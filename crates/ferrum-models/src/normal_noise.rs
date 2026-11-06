//! One explicit-state, weightless normal-noise tensor component.
//! Seed/offset are runtime input, not hidden global state or model weights.
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

/// The generator writes this precision directly. F32 never means a promoted
/// BF16 draw. The default retains existing Qwen configuration and sequence.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormalNoisePrecision {
    #[default]
    Bf16,
    F32,
}
impl NormalNoisePrecision {
    fn is_default(&self) -> bool {
        *self == Self::Bf16
    }
    fn element_type(self) -> ElementType {
        match self {
            Self::Bf16 => ElementType::Bf16,
            Self::F32 => ElementType::F32,
        }
    }
    fn operation(self) -> &'static str {
        match self {
            Self::Bf16 => PHILOX_NORMAL_BF16_OPERATION_ID,
            Self::F32 => PHILOX_NORMAL_F32_OPERATION_ID,
        }
    }
    fn family(self) -> &'static str {
        match self {
            Self::Bf16 => "family.normal_noise.philox.bf16",
            Self::F32 => "family.normal_noise.philox.f32",
        }
    }
    fn metadata(self) -> &'static str {
        match self {
            Self::Bf16 => "metadata.normal_noise.philox.bf16",
            Self::F32 => "metadata.normal_noise.philox.f32",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NormalNoiseConfig {
    channels: u64,
    height: u64,
    width: u64,
    #[serde(default, skip_serializing_if = "NormalNoisePrecision::is_default")]
    precision: NormalNoisePrecision,
}
impl NormalNoiseConfig {
    pub fn new(channels: u64, height: u64, width: u64) -> Result<Self, VNextError> {
        Self::new_with_precision(channels, height, width, NormalNoisePrecision::Bf16)
    }
    pub fn new_with_precision(
        channels: u64,
        height: u64,
        width: u64,
        precision: NormalNoisePrecision,
    ) -> Result<Self, VNextError> {
        let config = Self {
            channels,
            height,
            width,
            precision,
        };
        config.validate()?;
        Ok(config)
    }
    pub const fn precision(&self) -> NormalNoisePrecision {
        self.precision
    }
    pub fn state_tensor() -> Result<ResolvedTensorSpec, VNextError> {
        ResolvedTensorSpec::new(vec![4], ElementType::U32, ResolvedTensorLayout::Contiguous)
    }
    pub fn output_tensor(&self) -> Result<ResolvedTensorSpec, VNextError> {
        ResolvedTensorSpec::new(
            vec![1, self.channels, self.height, self.width],
            self.precision.element_type(),
            ResolvedTensorLayout::Contiguous,
        )
    }
    fn validate(&self) -> Result<(), VNextError> {
        match self.precision {
            NormalNoisePrecision::Bf16 => {
                PhiloxNormalBf16Geometry::new(&Self::state_tensor()?, &self.output_tensor()?)
                    .map(|_| ())
            }
            NormalNoisePrecision::F32 => {
                PhiloxNormalF32Geometry::new(&Self::state_tensor()?, &self.output_tensor()?)
                    .map(|_| ())
            }
        }
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
    precision: NormalNoisePrecision,
}
impl NormalNoiseFamily {
    pub const STATE_ID: &str = "value.normal_noise.seed_offset";
    pub const OUTPUT_ID: &str = "value.normal_noise.nchw";
    pub fn new(profile: &str) -> Result<Self, VNextError> {
        Self::new_with_precision(NormalNoisePrecision::Bf16, profile)
    }
    pub fn new_with_precision(
        precision: NormalNoisePrecision,
        profile: &str,
    ) -> Result<Self, VNextError> {
        Ok(Self {
            id: id(precision.family())?,
            profile: NumericalProfileId::new(profile).map_err(|reason| invalid(reason))?,
            precision,
        })
    }
}
impl ModelFamilyProvider for NormalNoiseFamily {
    type Config = NormalNoiseConfig;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id(self.precision.metadata()).expect("constant metadata ID")])
    }
    fn validate_config_identity(
        &self,
        raw: &Value,
        config: &NormalNoiseConfig,
    ) -> Result<(), VNextError> {
        config.validate()?;
        if config.precision != self.precision {
            return Err(invalid(
                "noise family precision must match the explicit config",
            ));
        }
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
        id(self.precision.metadata())
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
        config: &NormalNoiseConfig,
    ) -> Result<FamilyNumericalProfiles, VNextError> {
        config.validate()?;
        if config.precision != self.precision {
            return Err(invalid("noise family precision mismatch"));
        }
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: self.profile.clone(),
                family_id: self.id.clone(),
                version: ContractVersion::new(1, 0),
                kv_storage: vec![],
                primary_activation: id(Self::OUTPUT_ID)?,
                boundaries: BTreeMap::from([(
                    id(Self::OUTPUT_ID)?,
                    config.precision.element_type(),
                )]),
                states: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(config.precision.operation())?,
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
        if c.precision != self.precision {
            return Err(invalid("noise family precision mismatch"));
        }
        ModelProgram::new(
            self.id.clone(),
            vec![id(Self::STATE_ID)?],
            vec![ProgramBlock {
                id: "block.normal_noise".into(),
                nodes: vec![ProgramNode {
                    id: id("node.normal_noise.philox")?,
                    operation_id: id(c.precision.operation())?,
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

#[cfg(test)]
#[path = "../tests/support/philox_normal_oracle.rs"]
mod oracle;
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

    #[test]
    fn default_canonical_bf16_family_is_unchanged_and_f32_is_an_explicit_direct_operation() {
        let bf16 = NormalNoiseConfig::new(3, 1, 257).unwrap();
        let old = serde_json::json!({"channels":3,"height":1,"width":257});
        assert_eq!(serde_json::to_value(&bf16).unwrap(), old);
        assert_eq!(
            serde_json::from_value::<NormalNoiseConfig>(old.clone()).unwrap(),
            bf16
        );
        let bfamily = NormalNoiseFamily::new("numerical.test.noise.bf16").unwrap();
        assert_eq!(
            bfamily.family_id().as_str(),
            "family.normal_noise.philox.bf16"
        );
        assert_eq!(
            bfamily
                .validated_external_metadata_id(&old, &bf16)
                .unwrap()
                .as_str(),
            "metadata.normal_noise.philox.bf16"
        );
        let f32 =
            NormalNoiseConfig::new_with_precision(3, 1, 257, NormalNoisePrecision::F32).unwrap();
        assert_eq!(f32.precision(), NormalNoisePrecision::F32);
        assert_eq!(
            f32.output_tensor()
                .unwrap()
                .minimum_storage_bytes()
                .unwrap(),
            3084
        );
        let raw = serde_json::to_value(&f32).unwrap();
        assert_eq!(raw["precision"], "f32");
        assert!(bfamily.parse_config(&raw).is_err());
        let ffamily = NormalNoiseFamily::new_with_precision(
            NormalNoisePrecision::F32,
            "numerical.test.noise.f32",
        )
        .unwrap();
        assert!(ffamily.parse_config(&old).is_err());
        let prepared = TypedFamilyRegistration::new(ffamily)
            .prepare_with_profile(
                &raw,
                &NumericalProfileId::new("numerical.test.noise.f32").unwrap(),
            )
            .unwrap();
        assert_eq!(
            prepared.family_id().as_str(),
            "family.normal_noise.philox.f32"
        );
        assert_eq!(
            prepared.program().blocks()[0].nodes[0]
                .operation_id
                .as_str(),
            PHILOX_NORMAL_F32_OPERATION_ID
        );
        assert_eq!(
            prepared.numerical_profile().boundaries
                [&id::<ProgramValueId>(NormalNoiseFamily::OUTPUT_ID).unwrap()],
            ElementType::F32
        );
        assert_eq!(
            f32.compile_options().unwrap().tensor_specs()
                [&id::<ProgramValueId>(NormalNoiseFamily::OUTPUT_ID).unwrap()]
                .element_type,
            ElementType::F32
        );
        assert!(matches!(prepared.weights(), ModelWeights::None));
        // Four-byte output capacity is checked, not borrowed from BF16's
        // two-byte indexing limit. No payload or backend allocation is made.
        assert!(
            NormalNoiseConfig::new_with_precision(1, 1, 1 << 29, NormalNoisePrecision::F32)
                .is_err()
        );
        assert!(NormalNoiseConfig::new(1, 1, 1 << 29).is_ok());
    }
}
