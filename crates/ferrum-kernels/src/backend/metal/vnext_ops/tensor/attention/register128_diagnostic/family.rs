use super::*;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Visibility {
    AllValidBidirectional,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Mode {
    #[default]
    QkSoftmaxState,
    ValueWidth64,
    ValueWidth128,
}
impl Mode {
    pub(super) fn is_qk(&self) -> bool {
        *self == Self::QkSoftmaxState
    }
    pub(super) fn columns(self) -> u64 {
        match self {
            Self::QkSoftmaxState => 2,
            Self::ValueWidth64 => 64,
            Self::ValueWidth128 => 128,
        }
    }
    pub(super) fn inputs(self) -> &'static [&'static str] {
        if self.is_qk() {
            &[QUERY, KEY]
        } else {
            &[QUERY, KEY, VALUE]
        }
    }
    pub(super) fn operation(self) -> &'static str {
        if self.is_qk() {
            OP
        } else {
            VALUE_OP
        }
    }
    pub(super) fn output(self) -> &'static str {
        if self.is_qk() {
            OUTPUT
        } else {
            VALUE_OUTPUT
        }
    }
    pub(super) fn profile(self) -> &'static str {
        if self.is_qk() {
            PROFILE
        } else {
            VALUE_PROFILE
        }
    }
    pub(super) fn kernel(self) -> &'static str {
        match self {
            Self::QkSoftmaxState => "ferrum_tensor_attention_qk_prefix_f32",
            Self::ValueWidth64 => "ferrum_tensor_attention_values64_f32",
            Self::ValueWidth128 => "ferrum_tensor_attention_values128_f32",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Config {
    pub(super) batch: u64,
    pub(super) sequence: u64,
    pub(super) heads: u64,
    pub(super) visibility: Visibility,
    #[serde(default, skip_serializing_if = "Mode::is_qk")]
    pub(super) mode: Mode,
}

impl Config {
    pub(super) fn validate(self) -> Result<Self, VNextError> {
        if self.batch == 0 || self.batch > 16 || self.heads == 0 {
            return Err(invalid(
                "attention diagnostic requires batch1..16 and nonzero heads",
            ));
        }
        let rows = self
            .batch
            .checked_mul(self.sequence)
            .ok_or_else(|| invalid("row count overflow"))?;
        let columns = self
            .heads
            .checked_mul(128)
            .ok_or_else(|| invalid("head width overflow"))?;
        rows.checked_mul(columns)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| invalid("input byte extent overflow"))?;
        rows.checked_mul(self.heads)
            .and_then(|n| n.checked_mul(self.mode.columns()))
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| invalid("output byte extent overflow"))?;
        RegisterParams::new(
            self.batch
                .checked_mul(self.heads)
                .ok_or_else(|| invalid("head count overflow"))?,
            self.sequence,
            self.sequence,
            self.heads,
            self.heads,
            0.125,
        )
        .map_err(invalid)?;
        for spec in [self.input_spec(), self.output_spec()] {
            let bytes = ResolvedTensorSpec::new(spec.dimensions, spec.element_type, spec.layout)?
                .minimum_storage_bytes()?;
            if bytes > isize::MAX as u64 {
                return Err(invalid("diagnostic tensor exceeds address space"));
            }
        }
        Ok(self)
    }

    pub(super) fn input_spec(self) -> ProgramTensorSpec {
        ProgramTensorSpec {
            dimensions: vec![self.batch * self.sequence, self.heads * 128],
            element_type: ElementType::Bf16,
            layout: ResolvedTensorLayout::Contiguous,
        }
    }

    pub(super) fn output_spec(self) -> ProgramTensorSpec {
        ProgramTensorSpec {
            dimensions: vec![self.batch * self.sequence, self.heads, self.mode.columns()],
            element_type: ElementType::F32,
            layout: ResolvedTensorLayout::Contiguous,
        }
    }
}

pub(super) struct Family;
impl ModelFamilyProvider for Family {
    type Config = Config;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.metal.attention_diagnostic"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.metal.attention_diagnostic")])
    }
    fn validate_config_identity(&self, raw: &Value, config: &Config) -> Result<(), VNextError> {
        config.validate()?;
        if serde_json::to_value(config).unwrap() != *raw {
            return Err(invalid("noncanonical attention diagnostic configuration"));
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("metadata.metal.attention_diagnostic"))
    }
    fn parse_config(&self, raw: &Value) -> Result<Config, VNextError> {
        let config: Config =
            serde_json::from_value(raw.clone()).map_err(|e| invalid(e.to_string()))?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }
    fn weights(&self, _: &Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
    }
    fn numerical_profiles(&self, config: &Config) -> Result<FamilyNumericalProfiles, VNextError> {
        let profile = id::<NumericalProfileId>(config.mode.profile());
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: profile.clone(),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: id(config.mode.output()),
                boundaries: BTreeMap::from([(id(config.mode.output()), ElementType::F32)]),
                kv_storage: vec![],
                states: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(config.mode.operation()),
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
        config: &Config,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        ModelProgram::new(
            self.family_id().clone(),
            config.mode.inputs().iter().map(|name| id(*name)).collect(),
            vec![ProgramBlock {
                id: "block.attention_diagnostic".into(),
                nodes: vec![ProgramNode {
                    id: id("node.attention_diagnostic"),
                    operation_id: id(config.mode.operation()),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: config.mode.inputs().iter().map(|name| id(*name)).collect(),
                    outputs: vec![id(config.mode.output())],
                    attributes: BTreeMap::new(),
                }],
            }],
            vec![],
            vec![],
            vec![id(config.mode.output())],
        )?
        .with_input_tensor_specs(
            config
                .mode
                .inputs()
                .iter()
                .map(|name| (id(*name), config.input_spec()))
                .collect(),
        )
    }
    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

#[test]
fn prefix_profile_has_real_f32_output_and_rejects_unsupported_masks() {
    let raw = serde_json::json!({"batch":2,"sequence":33,"heads":2,"visibility":"all_valid_bidirectional"});
    TypedFamilyRegistration::new(Family)
        .prepare_with_profile(&raw, &id(PROFILE))
        .unwrap();
    let mut causal = raw.clone();
    causal["visibility"] = serde_json::json!("causal");
    assert!(TypedFamilyRegistration::new(Family)
        .prepare_with_profile(&causal, &id(PROFILE))
        .is_err());
    let mut short = raw;
    short["sequence"] = serde_json::json!(31);
    assert!(TypedFamilyRegistration::new(Family)
        .prepare_with_profile(&short, &id(PROFILE))
        .is_err());
    short["sequence"] = serde_json::json!(32);
    short["heads"] = serde_json::json!(1_u64 << 60);
    assert!(TypedFamilyRegistration::new(Family)
        .prepare_with_profile(&short, &id(PROFILE))
        .is_err());
}

#[test]
fn value_width_profiles_declare_three_inputs_and_real_f32_boundaries() {
    for mode in [Mode::ValueWidth64, Mode::ValueWidth128] {
        let config = Config {
            batch: 2,
            sequence: 33,
            heads: 2,
            visibility: Visibility::AllValidBidirectional,
            mode,
        };
        let raw = serde_json::to_value(config).unwrap();
        TypedFamilyRegistration::new(Family)
            .prepare_with_profile(&raw, &id(mode.profile()))
            .unwrap();
        let profile = Family.numerical_profiles(&config).unwrap();
        assert_eq!(config.mode.inputs(), &[QUERY, KEY, VALUE]);
        assert_eq!(config.output_spec().dimensions, vec![66, 2, mode.columns()]);
        assert_eq!(config.output_spec().element_type, ElementType::F32);
        drop(profile);
        let mut invalid = raw.clone();
        invalid["mode"] = serde_json::json!("value_width96");
        assert!(Family.parse_config(&invalid).is_err());
        let huge = Config {
            heads: 1 << 60,
            ..config
        };
        assert!(huge.validate().is_err());
    }
}
