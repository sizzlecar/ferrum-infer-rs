use super::*;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Config {}
pub(super) struct Family;

impl ModelFamilyProvider for Family {
    type Config = Config;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.metal.coordinate_probe"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.metal.coordinate_probe")])
    }
    fn validate_config_identity(&self, raw: &Value, config: &Config) -> Result<(), VNextError> {
        if serde_json::to_value(config).unwrap() != *raw {
            return Err(invalid("noncanonical coordinate probe configuration"));
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("metadata.metal.coordinate_probe"))
    }
    fn parse_config(&self, raw: &Value) -> Result<Config, VNextError> {
        let config = serde_json::from_value(raw.clone()).map_err(|e| invalid(format!("{e}")))?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }
    fn weights(&self, _: &Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
    }
    fn numerical_profiles(&self, _: &Config) -> Result<FamilyNumericalProfiles, VNextError> {
        let profile = id::<NumericalProfileId>(PROFILE);
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: profile.clone(),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                // Only node outputs are numerical boundaries. The real F32
                // echo supplies the floating primary boundary; coordinates
                // remain an independent ordinary U32 output.
                primary_activation: id(TAG_ECHO),
                boundaries: BTreeMap::from([
                    (id(TAG_ECHO), ElementType::F32),
                    (id(OUTPUT), ElementType::U32),
                ]),
                kv_storage: vec![],
                states: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(OP),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: None,
                    accumulation_type: None,
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                }],
            }],
            vec![profile],
        )
    }
    fn semantic_program(
        &self,
        _: &Config,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        ModelProgram::new(
            self.family_id().clone(),
            vec![id(INPUT)],
            vec![ProgramBlock {
                id: "block.coordinate_probe".into(),
                nodes: vec![ProgramNode {
                    id: id("node.coordinate_probe"),
                    operation_id: id(OP),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: vec![id(INPUT)],
                    outputs: vec![id(OUTPUT), id(TAG_ECHO)],
                    attributes: BTreeMap::new(),
                }],
            }],
            vec![],
            vec![],
            vec![id(OUTPUT), id(TAG_ECHO)],
        )?
        .with_input_tensor_specs(BTreeMap::from([(id(INPUT), input_spec())]))
    }
    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

#[test]
fn coordinate_probe_profile_covers_real_outputs_without_an_input_boundary() {
    // Exercise the production profile/program validator without a Metal device.
    // A floating input is not a legal substitute for a floating output boundary.
    TypedFamilyRegistration::new(Family)
        .prepare_with_profile(&serde_json::json!({}), &id(PROFILE))
        .unwrap();
}
