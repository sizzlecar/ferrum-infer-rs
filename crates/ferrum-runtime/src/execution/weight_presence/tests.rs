//! Weight presence is portable data; the trusted None witness additionally
//! binds a fully prepared family. All witnesses below use core constructors.

use super::{ExecutionWeights, TrustedExecutionWeights};
use crate::test_support::vnext_core_contract::{
    fixture_f32_profiles, id, TestConfig, TestFamily, TestRegistry,
};
use crate::*;
use serde_json::{json, Value};
use std::collections::BTreeSet;

struct ComponentFamily {
    node: &'static str,
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
        if config.width == 0 || serde_json::to_value(config).unwrap() != *raw {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "width".into(),
                reason: "component fixture requires a canonical positive width".into(),
            });
        }
        Ok(())
    }

    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &TestConfig,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("metadata.synthetic"))
    }

    fn parse_config(&self, raw: &Value) -> Result<TestConfig, VNextError> {
        let config = serde_json::from_value(raw.clone()).map_err(|error| {
            VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "config".into(),
                reason: error.to_string(),
            }
        })?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }

    fn weights(&self, _: &TestConfig) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
    }

    fn numerical_profiles(&self, _: &TestConfig) -> Result<FamilyNumericalProfiles, VNextError> {
        let profiles = fixture_f32_profiles(
            self.family_id(),
            &["value.output"],
            &["operation.main"],
            vec![],
        )?;
        let f32 = profiles.profiles()[0].clone();
        let mut bf16 = f32.clone();
        bf16.id = NumericalProfileId::new("fixture.bf16").unwrap();
        bf16.boundaries
            .insert(id("value.output"), ElementType::Bf16);
        FamilyNumericalProfiles::new(
            self.family_id(),
            profiles.version(),
            vec![f32.clone(), bf16],
            vec![f32.id],
        )
    }

    fn semantic_program(
        &self,
        _: &TestConfig,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        // Width is intentionally not embedded in this shape-polymorphic
        // semantic program, so the config-only rejection test isolates config.
        ModelProgram::new(
            self.family_id().clone(),
            vec![id("value.input")],
            vec![ProgramBlock {
                id: "block.main".into(),
                nodes: vec![ProgramNode {
                    id: id(self.node),
                    operation_id: id("operation.main"),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: vec![id("value.input")],
                    outputs: vec![id("value.output")],
                    attributes: Default::default(),
                }],
            }],
            vec![],
            vec![],
            vec![id("value.output")],
        )
    }

    fn semantic_metadata(&self, _: &TestConfig) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

fn prepared(width: u64, node: &'static str, profile: &str) -> PreparedModelFamily {
    TypedFamilyRegistration::new(ComponentFamily { node })
        .prepare_with_profile(
            &json!({"width": width}),
            &NumericalProfileId::new(profile).unwrap(),
        )
        .unwrap()
}

#[test]
fn trusted_none_binds_config_program_and_profile_not_just_family_id() {
    let original = prepared(4, "node.main", "fixture.f32");
    let config = prepared(8, "node.main", "fixture.f32");
    let program = prepared(4, "node.alternate", "fixture.f32");
    let profile = prepared(4, "node.main", "fixture.bf16");

    // Each adversary changes one independent part of the prepared identity.
    assert_ne!(original.canonical_config(), config.canonical_config());
    assert_eq!(original.program(), config.program());
    assert_eq!(original.numerical_profile(), config.numerical_profile());
    assert_eq!(original.canonical_config(), program.canonical_config());
    assert_ne!(original.program(), program.program());
    assert_eq!(original.numerical_profile(), program.numerical_profile());
    assert_eq!(original.canonical_config(), profile.canonical_config());
    assert_eq!(original.program(), profile.program());
    assert_ne!(original.numerical_profile(), profile.numerical_profile());

    let authority = TrustedExecutionWeights::for_family(&original).unwrap();
    authority.validate_against_family(&original).unwrap();
    authority
        .validate_against_family(&original.clone())
        .unwrap();
    let declaration: ExecutionWeights =
        serde_json::from_value(serde_json::to_value(authority.declaration()).unwrap()).unwrap();
    assert!(declaration.is_none());

    for different in [&config, &program, &profile] {
        assert_eq!(original.family_id(), different.family_id());
        assert_eq!(original.weights(), different.weights());
        assert_ne!(
            original.fingerprint().unwrap(),
            different.fingerprint().unwrap()
        );
        // An untrusted absence declaration truthfully describes both families;
        // it does not transfer the process-local, original-family witness.
        declaration.validate_against_family(different).unwrap();
        assert!(authority.validate_against_family(different).is_err());
        let own = TrustedExecutionWeights::for_family(different).unwrap();
        own.validate_against_family(different).unwrap();
        assert_eq!(own.declaration(), declaration);
    }
}

#[test]
fn stored_and_none_authorities_cannot_cross_presence_with_the_same_family_id() {
    let none = prepared(4, "node.main", "fixture.f32");
    let stored = TestRegistry::new().prepare();
    assert_eq!(none.family_id(), stored.family_id());
    assert!(matches!(stored.weights(), ModelWeights::Stored(_)));
    let none_authority = TrustedExecutionWeights::for_family(&none).unwrap();
    let stored_authority = TrustedExecutionWeights::for_family(&stored).unwrap();
    none_authority.validate_against_family(&none).unwrap();
    stored_authority.validate_against_family(&stored).unwrap();
    assert!(none_authority.validate_against_family(&stored).is_err());
    assert!(stored_authority.validate_against_family(&none).is_err());
    assert!(none_authority
        .declaration()
        .validate_against_family(&stored)
        .is_err());
    assert!(stored_authority
        .declaration()
        .validate_against_family(&none)
        .is_err());
}
