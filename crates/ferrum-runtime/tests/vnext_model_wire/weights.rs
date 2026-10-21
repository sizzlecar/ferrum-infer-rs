use super::*;

#[derive(Clone, Copy)]
enum WeightCase {
    None,
    Stored,
    EmptyStored,
}

struct ComponentFamily {
    weights: WeightCase,
    declare_weight: bool,
    specialization_calls: Arc<AtomicUsize>,
}

impl ModelFamilyProvider for ComponentFamily {
    type Config = TestConfig;

    fn family_id(&self) -> &ModelFamilyId {
        TestFamily.family_id()
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        TestFamily.external_metadata_ids()
    }

    fn validate_config_identity(&self, raw: &Value, config: &TestConfig) -> Result<(), VNextError> {
        TestFamily.validate_config_identity(raw, config)
    }

    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &TestConfig,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        TestFamily.validated_external_metadata_id(raw, config)
    }

    fn parse_config(&self, raw: &Value) -> Result<TestConfig, VNextError> {
        TestFamily.parse_config(raw)
    }

    fn weights(&self, config: &TestConfig) -> Result<ModelWeights, VNextError> {
        match self.weights {
            WeightCase::None => Ok(ModelWeights::None),
            WeightCase::Stored => TestFamily.weights(config),
            WeightCase::EmptyStored => {
                let mut schema = TestFamily.weights(config)?.stored_schema()?.clone();
                schema.components.clear();
                schema.tensors.clear();
                Ok(ModelWeights::Stored(schema))
            }
        }
    }

    fn numerical_profiles(&self, _: &TestConfig) -> Result<FamilyNumericalProfiles, VNextError> {
        let profiles = fixture_f32_profiles(
            self.family_id(),
            &["value.output"],
            &["operation.model-wire-test"],
            Vec::new(),
        )?;
        let first = profiles.profiles()[0].clone();
        let mut second = first.clone();
        second.id = NumericalProfileId::new("fixture.bf16").unwrap();
        second
            .boundaries
            .insert(id("value.output"), ElementType::Bf16);
        FamilyNumericalProfiles::new(
            self.family_id(),
            profiles.version(),
            vec![first.clone(), second],
            vec![first.id],
        )
    }

    fn specialize_weight_schema(
        &self,
        _: &TestConfig,
        source: &WeightSchema,
        _: &NumericalExecutionProfile,
    ) -> Result<WeightSchema, VNextError> {
        self.specialization_calls.fetch_add(1, Ordering::SeqCst);
        if matches!(self.weights, WeightCase::None) {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "specialization".to_owned(),
                reason: "weightless family must not enter stored specialization".to_owned(),
            });
        }
        Ok(source.clone())
    }

    fn semantic_program(
        &self,
        config: &TestConfig,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let mut inputs = vec![id("value.input")];
        let weights = if self.declare_weight {
            inputs.push(id("value.weight"));
            vec![WeightReference {
                weight_id: id("weight.tensor.model-wire-test"),
                value_id: id("value.weight"),
                tensor: ProgramTensorSpec {
                    dimensions: vec![config.width],
                    element_type: ElementType::F32,
                    layout: ResolvedTensorLayout::Contiguous,
                },
            }]
        } else {
            Vec::new()
        };
        ModelProgram::new(
            self.family_id().clone(),
            vec![id("value.input")],
            vec![ProgramBlock {
                id: "block.main".to_owned(),
                nodes: vec![ProgramNode {
                    id: id("node.main"),
                    operation_id: id("operation.model-wire-test"),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs,
                    outputs: vec![id("value.output")],
                    attributes: BTreeMap::new(),
                }],
            }],
            Vec::new(),
            weights,
            vec![id("value.output")],
        )
    }

    fn semantic_metadata(&self, config: &TestConfig) -> Result<ModelSemanticMetadata, VNextError> {
        TestFamily.semantic_metadata(config)
    }
}

struct ComponentRegistry {
    registration: TypedFamilyRegistration<ComponentFamily>,
    specialization_calls: Arc<AtomicUsize>,
}

impl ComponentRegistry {
    fn new(weights: WeightCase, declare_weight: bool) -> Self {
        let specialization_calls = Arc::new(AtomicUsize::new(0));
        Self {
            registration: TypedFamilyRegistration::new(ComponentFamily {
                weights,
                declare_weight,
                specialization_calls: Arc::clone(&specialization_calls),
            }),
            specialization_calls,
        }
    }
}

impl ModelFamilyRegistry for ComponentRegistry {
    fn registrations(&self) -> Vec<&dyn ModelFamilyRegistration> {
        vec![&self.registration]
    }
}

fn tensor_config() -> Value {
    json!({
        "model_type": "metadata.model-wire-test",
        "width": 4,
        "metadata": { "kind": "tensor_component" }
    })
}

#[test]
fn weightless_component_preserves_absence_across_profiles_and_wire_rebuild() {
    let registry = ComponentRegistry::new(WeightCase::None, false);
    let definition = registry.registration.define(&tensor_config()).unwrap();
    assert_eq!(definition.weights(), &ModelWeights::None);
    assert!(definition.stored_weight_schema().is_err());
    let mut fingerprints = BTreeSet::new();
    for profile in definition.numerical_profiles().profiles() {
        let prepared = registry
            .registration
            .prepare(&definition, &profile.id)
            .unwrap();
        assert_eq!(prepared.weights(), &ModelWeights::None);
        assert!(prepared.stored_weight_schema().is_err());
        assert!(prepared.program().weights().is_empty());
        let wire = serde_json::to_value(&prepared).unwrap();
        assert_eq!(wire["weights"], json!({ "kind": "none" }));
        assert_eq!(wire["wire_version"], json!(4));
        assert!(wire.get("weight_schema").is_none());
        let rebuilt = PreparedModelFamily::from_json_validated(
            &serde_json::to_vec(&wire).unwrap(),
            &registry,
        )
        .unwrap();
        assert_eq!(rebuilt, prepared);
        assert!(fingerprints.insert(prepared.fingerprint().unwrap()));
    }
    assert_eq!(registry.specialization_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn weight_presence_and_program_references_fail_closed() {
    let none = ComponentRegistry::new(WeightCase::None, false);
    let language = json!({ "model_type": "metadata.model-wire-test", "width": 4 });
    assert!(matches!(none.registration.define(&language),
        Err(VNextError::InvalidModelConfig { field, .. }) if field == "weights"));

    let with_reference = ComponentRegistry::new(WeightCase::None, true);
    assert!(
        matches!(with_reference.registration.prepare_fixture(&tensor_config()),
        Err(VNextError::InvalidModelConfig { field, .. }) if field == "program.weights")
    );

    let malformed = ComponentRegistry::new(WeightCase::EmptyStored, false);
    assert!(malformed.registration.define(&tensor_config()).is_err());

    let missing_required = ComponentRegistry::new(WeightCase::Stored, false);
    assert!(
        matches!(missing_required.registration.prepare_fixture(&tensor_config()),
        Err(VNextError::InvalidModelConfig { field, .. }) if field == "program.weights")
    );

    let stored = ComponentRegistry::new(WeightCase::Stored, true);
    let prepared = stored
        .registration
        .prepare_fixture(&tensor_config())
        .unwrap();
    assert!(matches!(prepared.weights(), ModelWeights::Stored(_)));
    assert_eq!(stored.specialization_calls.load(Ordering::SeqCst), 1);
    let wire = serde_json::to_value(&prepared).unwrap();
    assert_eq!(wire["weights"]["kind"], json!("stored"));
    assert_eq!(
        wire["weights"]["schema"],
        serde_json::to_value(prepared.stored_weight_schema().unwrap()).unwrap()
    );
    assert_eq!(
        PreparedModelFamily::from_json_validated(&serde_json::to_vec(&wire).unwrap(), &stored,)
            .unwrap(),
        prepared
    );
}

#[test]
fn model_weights_wire_rejects_cross_variant_fields_and_legacy_packages() {
    for invalid in [
        json!({}),
        json!({ "kind": "unknown" }),
        json!({ "kind": "stored" }),
        json!({ "kind": "none", "schema": null }),
        json!({ "kind": "none", "materializer_id": "identity" }),
        json!({ "kind": "none", "source_map": {} }),
        json!({ "kind": "none", "quality_approval": null }),
    ] {
        assert!(serde_json::from_value::<ModelWeights>(invalid).is_err());
    }

    let registry = ComponentRegistry::new(WeightCase::None, false);
    let prepared = registry
        .registration
        .prepare_fixture(&tensor_config())
        .unwrap();
    let wire = serde_json::to_value(&prepared).unwrap();
    for invalid_weights in [
        json!({ "kind": "none", "schema": {} }),
        json!({ "kind": "none", "unexpected": true }),
    ] {
        let mut changed = wire.clone();
        changed["weights"] = invalid_weights;
        assert!(
            PreparedModelFamily::decode_untrusted(&serde_json::to_vec(&changed).unwrap()).is_err()
        );
    }
    let mut legacy = wire.clone();
    legacy["wire_version"] = json!(3);
    assert!(PreparedModelFamily::decode_untrusted(&serde_json::to_vec(&legacy).unwrap()).is_err());
    let mut legacy_field = wire.clone();
    legacy_field["weight_schema"] = json!({});
    assert!(
        PreparedModelFamily::decode_untrusted(&serde_json::to_vec(&legacy_field).unwrap()).is_err()
    );

    let stored = ComponentRegistry::new(WeightCase::Stored, true);
    let stored_prepared = stored
        .registration
        .prepare_fixture(&tensor_config())
        .unwrap();
    let stored_wire = serde_json::to_value(stored_prepared).unwrap();
    let mut unknown_stored = stored_wire.clone();
    unknown_stored["weights"]["schema"]["caller_override"] = json!(true);
    assert!(
        PreparedModelFamily::decode_untrusted(&serde_json::to_vec(&unknown_stored).unwrap(),)
            .is_err()
    );
    let mut forged = stored_wire;
    forged["weights"] = json!({ "kind": "none" });
    assert!(PreparedModelFamily::from_json_validated(
        &serde_json::to_vec(&forged).unwrap(),
        &stored,
    )
    .is_err());
    let mut forged_stored = wire;
    forged_stored["weights"] =
        serde_json::to_value(TestRegistry::new().prepare().weights()).unwrap();
    assert!(PreparedModelFamily::from_json_validated(
        &serde_json::to_vec(&forged_stored).unwrap(),
        &registry,
    )
    .is_err());
}
