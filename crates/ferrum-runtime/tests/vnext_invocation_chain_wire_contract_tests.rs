mod vnext_core_contract;
use vnext_core_contract::*;

struct ForkFamily;

impl ModelFamilyProvider for ForkFamily {
    type Config = TestConfig;

    fn family_id(&self) -> &ModelFamilyId {
        SequentialScratchFamily.family_id()
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        SequentialScratchFamily.external_metadata_ids()
    }
    fn validate_config_identity(&self, raw: &Value, config: &TestConfig) -> Result<(), VNextError> {
        SequentialScratchFamily.validate_config_identity(raw, config)
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &TestConfig,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        SequentialScratchFamily.validated_external_metadata_id(raw, config)
    }
    fn parse_config(&self, raw: &Value) -> Result<TestConfig, VNextError> {
        SequentialScratchFamily.parse_config(raw)
    }
    fn weights(&self, config: &TestConfig) -> Result<ModelWeights, VNextError> {
        SequentialScratchFamily.weights(config)
    }
    fn numerical_profiles(
        &self,
        config: &TestConfig,
    ) -> Result<FamilyNumericalProfiles, VNextError> {
        fixture_f32_profiles(
            self.family_id(),
            &["value.a", "value.b", "value.c", "value.d"],
            &["operation.main"],
            vec![fixture_byte_state(config.width)],
        )
    }
    fn semantic_metadata(&self, config: &TestConfig) -> Result<ModelSemanticMetadata, VNextError> {
        SequentialScratchFamily.semantic_metadata(config)
    }
    fn semantic_program(
        &self,
        config: &TestConfig,
        profile: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let base = SequentialScratchFamily.semantic_program(config, profile)?;
        let nodes = [("a", "input"), ("b", "input"), ("c", "a"), ("d", "b")]
            .into_iter()
            .map(|(name, input)| ProgramNode {
                id: id(format!("node.{name}")),
                operation_id: id("operation.main"),
                required_version: ContractVersion::new(1, 0),
                work: ProgramNodeWorkSpec::Fixed,
                inputs: vec![
                    id(format!("value.{input}")),
                    id("value.weight"),
                    id("value.state"),
                ],
                outputs: vec![id(format!("value.{name}"))],
                attributes: BTreeMap::new(),
            })
            .collect();
        ModelProgram::new(
            self.family_id().clone(),
            base.inputs().to_vec(),
            vec![ProgramBlock {
                id: "block.fork".to_owned(),
                nodes,
            }],
            base.states().to_vec(),
            base.weights().to_vec(),
            vec![id("value.c"), id("value.d")],
        )
    }
}

struct Fixture {
    family: PreparedModelFamily,
    catalog: CapabilityCatalog,
    policy: ResolvedRuntimePolicy,
    _planning: TestPlanningRegistry,
    compilation: ProgramPlanCompilation,
}

impl Fixture {
    fn new() -> Self {
        let family = TypedFamilyRegistration::new(ForkFamily)
            .prepare_fixture(&json!({"width": 4}))
            .unwrap();
        let catalog = catalog();
        let policy = policy(4096);
        let planning = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
        let options = ProgramPlanCompileOptions::new(BTreeMap::from([(
            id("value.input"),
            ProgramTensorSpec {
                dimensions: vec![4],
                element_type: ElementType::F32,
                layout: ResolvedTensorLayout::Contiguous,
            },
        )]))
        .unwrap();
        let compilation = ProgramPlanCompiler::compile(
            &family,
            &catalog,
            &policy,
            &planning.planning(),
            &options,
        )
        .unwrap();
        Self {
            family,
            catalog,
            policy,
            _planning: planning,
            compilation,
        }
    }
    fn plan(&self) -> &ExecutionPlan {
        self.compilation.executable().execution_plan()
    }
    fn restore(&self, value: &Value) -> Result<ExecutionPlan, VNextError> {
        ExecutionPlan::decode_untrusted(&serde_json::to_vec(value).unwrap())?.revalidate(
            &self.family,
            &self.catalog,
            &self.policy,
            self.compilation.node_resolutions().to_vec(),
        )
    }
    fn validate_against(&self, plan: &ExecutionPlan) -> Result<(), VNextError> {
        plan.validate_against(
            &self.family,
            &self.catalog,
            &self.policy,
            self.compilation.node_resolutions(),
        )
    }
    fn legacy_wire(&self) -> Value {
        let mut value = serde_json::to_value(self.plan()).unwrap();
        let memory = &mut value["payload"]["memory"];
        let mut extra = 0;
        let mut changed = false;
        for pool in memory["dynamic_pools"].as_array_mut().unwrap() {
            if pool["invocation_liveness_mode"] != json!("dependency_chain_reuse") {
                continue;
            }
            changed = true;
            let old_peak = pool["minimum_invocation_peak_bytes"].as_u64().unwrap();
            // This provider declares one fixed 64-byte scratch per node.
            let sum = 64 * pool["invocation_liveness"].as_array().unwrap().len() as u64;
            extra += sum - old_peak;
            pool["minimum_invocation_peak_bytes"] = json!(sum);
            pool["provisioning"]["minimum_resident_bytes"] = json!(sum);
            pool["invocation_liveness_mode"] = json!("conservative_concurrent");
            for row in pool["invocation_liveness"].as_array_mut().unwrap() {
                row.as_object_mut().unwrap().remove("chain_slot");
            }
        }
        assert!(changed);
        for field in [
            "minimum_invocation_peak_bytes",
            "minimum_runnable_request_bytes",
        ] {
            memory[field] = json!(memory[field].as_u64().unwrap() + extra);
        }
        rehash_plan_json(&mut value);
        value
    }
}

#[test]
fn invocation_chain_public_wire_revalidation_preserves_legacy_hash_and_layout() {
    let fixture = Fixture::new();
    let legacy = fixture.legacy_wire();
    let restored = fixture.restore(&legacy).unwrap();
    assert_eq!(serde_json::to_value(&restored).unwrap(), legacy);
    assert_ne!(restored.plan_hash(), fixture.plan().plan_hash());
    assert_eq!(
        restored.payload().memory().minimum_invocation_peak_bytes(),
        256
    );
    fixture.validate_against(&restored).unwrap();
    let again = ExecutionPlan::decode_untrusted(&restored.to_json().unwrap())
        .unwrap()
        .revalidate(
            &fixture.family,
            &fixture.catalog,
            &fixture.policy,
            fixture.compilation.node_resolutions().to_vec(),
        )
        .unwrap();
    assert_eq!(again.to_json().unwrap(), restored.to_json().unwrap());
}

#[test]
fn invocation_chain_new_wire_revalidates_and_rehashed_alias_or_dependency_changes_fail() {
    let fixture = Fixture::new();
    let wire = serde_json::to_value(fixture.plan()).unwrap();
    let restored = fixture.restore(&wire).unwrap();
    assert_eq!(&restored, fixture.plan());
    fixture.validate_against(&restored).unwrap();
    let mut merged = wire.clone();
    let memory = &mut merged["payload"]["memory"];
    for pool in memory["dynamic_pools"].as_array_mut().unwrap() {
        if pool["invocation_liveness_mode"] != json!("dependency_chain_reuse") {
            continue;
        }
        assert_eq!(pool["minimum_invocation_peak_bytes"], json!(128));
        for row in pool["invocation_liveness"].as_array_mut().unwrap() {
            row["chain_slot"] = json!(0);
        }
        pool["minimum_invocation_peak_bytes"] = json!(64);
        pool["provisioning"]["minimum_resident_bytes"] = json!(64);
    }
    for field in [
        "minimum_invocation_peak_bytes",
        "minimum_runnable_request_bytes",
    ] {
        memory[field] = json!(memory[field].as_u64().unwrap() - 64);
    }
    rehash_plan_json(&mut merged);
    // Structurally coherent and correctly rehashed, but aliases parallel rows.
    assert!(ExecutionPlan::decode_untrusted(&serde_json::to_vec(&merged).unwrap()).is_ok());
    assert!(fixture.restore(&merged).is_err());
    let mut dependency = wire;
    dependency["payload"]["nodes"][2]["dependencies"] = json!([]);
    rehash_plan_json(&mut dependency);
    assert!(fixture.restore(&dependency).is_err());
}

#[test]
fn invocation_chain_legacy_compatibility_does_not_adopt_other_untrusted_fields() {
    let fixture = Fixture::new();
    let mut wire = fixture.legacy_wire();
    wire["payload"]["policy_fingerprint"] = json!(sha('0'));
    rehash_plan_json(&mut wire);
    assert!(fixture.restore(&wire).is_err());
    let mut wire = fixture.legacy_wire();
    wire["payload"]["memory"]["dynamic_pools"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|pool| pool["invocation_liveness_mode"] == json!("conservative_concurrent"))
        .unwrap()["invocation_liveness"][0]["chain_slot"] = json!(0);
    rehash_plan_json(&mut wire);
    assert!(fixture.restore(&wire).is_err());
}
