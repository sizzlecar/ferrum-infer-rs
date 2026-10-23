//! One explicit-state, weightless normal-noise tensor component.
//! Seed/offset are runtime input, not hidden global state or model weights.
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NormalNoiseConfig {
    channels: u64,
    height: u64,
    width: u64,
}
impl NormalNoiseConfig {
    pub fn new(channels: u64, height: u64, width: u64) -> Result<Self, VNextError> {
        let config = Self {
            channels,
            height,
            width,
        };
        config.validate()?;
        Ok(config)
    }
    pub fn state_tensor() -> Result<ResolvedTensorSpec, VNextError> {
        ResolvedTensorSpec::new(vec![4], ElementType::U32, ResolvedTensorLayout::Contiguous)
    }
    pub fn output_tensor(&self) -> Result<ResolvedTensorSpec, VNextError> {
        ResolvedTensorSpec::new(
            vec![1, self.channels, self.height, self.width],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
    }
    fn validate(&self) -> Result<(), VNextError> {
        PhiloxNormalBf16Geometry::new(&Self::state_tensor()?, &self.output_tensor()?).map(|_| ())
    }
    pub fn compile_options(&self) -> Result<ProgramPlanCompileOptions, VNextError> {
        self.validate()?;
        let mut options = ProgramPlanCompileOptions::new(BTreeMap::from([(
            id(NormalNoiseFamily::STATE_ID)?,
            spec(Self::state_tensor()?),
        )]))?;
        options.insert_tensor_spec(
            id(NormalNoiseFamily::OUTPUT_ID)?,
            spec(self.output_tensor()?),
        )?;
        options.retain_completion_value(id(NormalNoiseFamily::OUTPUT_ID)?);
        Ok(options)
    }
}

pub struct NormalNoiseFamily {
    id: ModelFamilyId,
    profile: NumericalProfileId,
}
impl NormalNoiseFamily {
    pub const STATE_ID: &str = "value.normal_noise.seed_offset";
    pub const OUTPUT_ID: &str = "value.normal_noise.nchw";
    pub fn new(profile: &str) -> Result<Self, VNextError> {
        Ok(Self {
            id: id("family.normal_noise.philox.bf16")?,
            profile: NumericalProfileId::new(profile).map_err(|reason| invalid(reason))?,
        })
    }
}
impl ModelFamilyProvider for NormalNoiseFamily {
    type Config = NormalNoiseConfig;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.normal_noise.philox.bf16").expect("constant metadata ID")])
    }
    fn validate_config_identity(
        &self,
        raw: &Value,
        config: &NormalNoiseConfig,
    ) -> Result<(), VNextError> {
        config.validate()?;
        if serde_json::to_value(config).map_err(|e| invalid(e.to_string()))? != *raw {
            return Err(invalid("noise config must be canonical"));
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &NormalNoiseConfig,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        id("metadata.normal_noise.philox.bf16")
    }
    fn parse_config(&self, raw: &Value) -> Result<NormalNoiseConfig, VNextError> {
        let c = serde_json::from_value(raw.clone()).map_err(|e| invalid(e.to_string()))?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn weights(&self, _: &NormalNoiseConfig) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
    }
    fn numerical_profiles(
        &self,
        _: &NormalNoiseConfig,
    ) -> Result<FamilyNumericalProfiles, VNextError> {
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: self.profile.clone(),
                family_id: self.id.clone(),
                version: ContractVersion::new(1, 0),
                kv_storage: vec![],
                primary_activation: id(Self::OUTPUT_ID)?,
                boundaries: BTreeMap::from([(id(Self::OUTPUT_ID)?, ElementType::Bf16)]),
                states: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(PHILOX_NORMAL_BF16_OPERATION_ID)?,
                    version: ContractVersion::new(1, 0),
                    multiplication_type: Some(ElementType::F32),
                    accumulation_type: None,
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                }],
            }],
            vec![self.profile.clone()],
        )
    }
    fn semantic_program(
        &self,
        c: &NormalNoiseConfig,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        c.validate()?;
        ModelProgram::new(
            self.id.clone(),
            vec![id(Self::STATE_ID)?],
            vec![ProgramBlock {
                id: "block.normal_noise".into(),
                nodes: vec![ProgramNode {
                    id: id("node.normal_noise.philox")?,
                    operation_id: id(PHILOX_NORMAL_BF16_OPERATION_ID)?,
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: vec![id(Self::STATE_ID)?],
                    outputs: vec![id(Self::OUTPUT_ID)?],
                    attributes: BTreeMap::new(),
                }],
            }],
            vec![],
            vec![],
            vec![id(Self::OUTPUT_ID)?],
        )?
        .with_input_tensor_specs(BTreeMap::from([(
            id(Self::STATE_ID)?,
            spec(NormalNoiseConfig::state_tensor()?),
        )]))
    }
    fn semantic_metadata(
        &self,
        _: &NormalNoiseConfig,
    ) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn spec(t: ResolvedTensorSpec) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions: t.dimensions().to_vec(),
        element_type: t.element_type(),
        layout: t.layout().clone(),
    }
}
fn id<T: TryFrom<String, Error = VNextError>>(s: &str) -> Result<T, VNextError> {
    T::try_from(s.to_owned())
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_shape_state_and_no_weights_survive_preparation() {
        let c = NormalNoiseConfig::new(64, 2, 3).unwrap();
        let family = TypedFamilyRegistration::new(
            NormalNoiseFamily::new("numerical.test.philox.unqualified").unwrap(),
        )
        .prepare_with_profile(
            &serde_json::to_value(&c).unwrap(),
            &NumericalProfileId::new("numerical.test.philox.unqualified").unwrap(),
        )
        .unwrap();
        assert!(matches!(family.weights(), ModelWeights::None));
        assert_eq!(c.output_tensor().unwrap().dimensions(), [1, 64, 2, 3]);
        let options = c.compile_options().unwrap();
        assert_eq!(options.tensor_specs().len(), 2);
        assert_eq!(
            options
                .tensor_specs()
                .get(&id(NormalNoiseFamily::STATE_ID).unwrap())
                .unwrap()
                .element_type,
            ElementType::U32
        );
        assert!(NormalNoiseConfig::new(0, 2, 3).is_err());
        assert!(NormalNoiseConfig::new(64, 1 << 30, 1).is_err());
    }
}
