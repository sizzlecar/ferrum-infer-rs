use super::*;

// Reuse the existing typed source/program fixture. Only its declared output
// storage differs; no numerical kernel or fake floating output is introduced.
struct IntegerPrimaryFamily;
impl ModelFamilyProvider for IntegerPrimaryFamily {
    type Config = TestConfig;
    fn family_id(&self) -> &ModelFamilyId {
        TestFamily.family_id()
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        TestFamily.external_metadata_ids()
    }
    fn validate_config_identity(&self, raw: &Value, c: &TestConfig) -> Result<(), VNextError> {
        TestFamily.validate_config_identity(raw, c)
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        c: &TestConfig,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        TestFamily.validated_external_metadata_id(raw, c)
    }
    fn parse_config(&self, raw: &Value) -> Result<TestConfig, VNextError> {
        TestFamily.parse_config(raw)
    }
    fn weights(&self, c: &TestConfig) -> Result<ModelWeights, VNextError> {
        TestFamily.weights(c)
    }
    fn numerical_profiles(&self, c: &TestConfig) -> Result<FamilyNumericalProfiles, VNextError> {
        let mut profile = TestFamily.numerical_profiles(c)?.profiles()[0].clone();
        profile.id = NumericalProfileId::new("fixture.packed-storage").unwrap();
        profile
            .boundaries
            .insert(id("value.output"), ElementType::U8);
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![profile],
            vec![NumericalProfileId::new("fixture.packed-storage").unwrap()],
        )
    }
    fn semantic_program(
        &self,
        c: &TestConfig,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        TestFamily.semantic_program(c, p)
    }
    fn semantic_metadata(&self, c: &TestConfig) -> Result<ModelSemanticMetadata, VNextError> {
        TestFamily.semantic_metadata(c)
    }
}
struct IntegerRegistry(TypedFamilyRegistration<IntegerPrimaryFamily>);
impl ModelFamilyRegistry for IntegerRegistry {
    fn registrations(&self) -> Vec<&dyn ModelFamilyRegistration> {
        vec![&self.0]
    }
}
fn config(tensor: bool) -> Value {
    let mut config = json!({"model_type":"metadata.model-wire-test", "width":4});
    if tensor {
        config["metadata"] = json!({"kind":"tensor_component"});
    }
    config
}

#[test]
fn tensor_storage_primary_prepares_and_revalidates_without_weakening_its_boundaries() {
    let registry = IntegerRegistry(TypedFamilyRegistration::new(IntegerPrimaryFamily));
    let prepared = registry
        .0
        .prepare_with_profile(
            &config(true),
            &NumericalProfileId::new("fixture.packed-storage").unwrap(),
        )
        .unwrap();
    let profile = prepared.numerical_profile();
    assert_eq!(profile.primary_storage_type().unwrap(), ElementType::U8);
    assert!(profile.activation_type().is_err());
    assert_eq!(
        prepared.program().outputs(),
        [id::<ProgramValueId>("value.output")]
    );
    let bytes = serde_json::to_vec(&prepared).unwrap();
    assert_eq!(
        PreparedModelFamily::from_json_validated(&bytes, &registry).unwrap(),
        prepared
    );

    let inferred = |dtype| {
        BTreeMap::from([(
            id("value.output"),
            ResolvedTensorSpec::new(vec![4], dtype, ResolvedTensorLayout::Contiguous).unwrap(),
        )])
    };
    profile
        .validate_inferred_boundaries(&inferred(ElementType::U8))
        .unwrap();
    assert!(profile
        .validate_inferred_boundaries(&inferred(ElementType::F32))
        .is_err());
    let mut missing_primary = profile.clone();
    missing_primary.primary_activation = id("value.input");
    assert!(missing_primary.validate().is_err());
    let mut extra_boundary = profile.clone();
    extra_boundary
        .boundaries
        .insert(id("value.input"), ElementType::F32);
    assert!(extra_boundary.validate_program(prepared.program()).is_err());

    let mut changed = serde_json::to_value(&prepared).unwrap();
    changed["numerical_profile"]["boundaries"]["value.output"] = json!("f32");
    assert!(PreparedModelFamily::from_json_validated(
        &serde_json::to_vec(&changed).unwrap(),
        &registry
    )
    .is_err());
}

#[test]
fn causal_definition_rejects_integer_primary_before_program_or_product_adapters() {
    let registry = IntegerRegistry(TypedFamilyRegistration::new(IntegerPrimaryFamily));
    for error in [
        registry.0.define(&config(false)).err().unwrap(),
        registry
            .0
            .prepare_with_profile(
                &config(false),
                &NumericalProfileId::new("fixture.packed-storage").unwrap(),
            )
            .err()
            .unwrap(),
    ] {
        assert!(
            matches!(error, VNextError::InvalidModelConfig { field, reason, .. }
            if field == "numerical_profile" && reason.contains("causal language primary activation"))
        );
    }
    let prepared = registry
        .0
        .prepare_with_profile(
            &config(true),
            &NumericalProfileId::new("fixture.packed-storage").unwrap(),
        )
        .unwrap();
    let mut forged = serde_json::to_value(&prepared).unwrap();
    let causal = TestFamily
        .semantic_metadata(&TestFamily.parse_config(&config(false)).unwrap())
        .unwrap();
    forged["metadata"] = serde_json::to_value(causal).unwrap();
    assert!(PreparedModelFamily::from_json_validated(
        &serde_json::to_vec(&forged).unwrap(),
        &registry
    )
    .is_err());
}
