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
    ProductionBf16,
    ProductionNativeBf16,
    SteelReference,
    CompactDenseBf16,
    DecodeShift,
    DecodeNative,
}
impl Mode {
    pub(super) fn is_qk(&self) -> bool {
        *self == Self::QkSoftmaxState
    }
    pub(super) fn is_decode(self) -> bool {
        matches!(self, Self::DecodeShift | Self::DecodeNative)
    }
    pub(super) fn native_decode(self) -> bool {
        matches!(self, Self::ProductionNativeBf16 | Self::DecodeNative)
    }
    pub(super) fn explicit_msl31(self) -> bool {
        self.is_bf16() || self.is_decode()
    }
    pub(super) fn uses_dense_params(self) -> bool {
        matches!(self, Self::SteelReference | Self::CompactDenseBf16)
    }
    pub(super) fn columns(self) -> u64 {
        match self {
            Self::QkSoftmaxState => 2,
            Self::ValueWidth64 => 64,
            Self::ValueWidth128
            | Self::ProductionBf16
            | Self::ProductionNativeBf16
            | Self::SteelReference
            | Self::CompactDenseBf16
            | Self::DecodeShift
            | Self::DecodeNative => 128,
        }
    }
    pub(super) fn is_bf16(self) -> bool {
        matches!(
            self,
            Self::ProductionBf16
                | Self::ProductionNativeBf16
                | Self::SteelReference
                | Self::CompactDenseBf16
        )
    }
    pub(super) fn output_element(self) -> ElementType {
        if self.is_bf16() {
            ElementType::Bf16
        } else {
            ElementType::F32
        }
    }
    pub(super) fn dynamic_bytes(self) -> u64 {
        if self.uses_dense_params() || self.is_decode() {
            0
        } else {
            SHARED_BYTES
        }
    }
    pub(super) fn inputs(self) -> &'static [&'static str] {
        if self.is_decode() {
            &[QUERY]
        } else if self.is_qk() {
            &[QUERY, KEY]
        } else {
            &[QUERY, KEY, VALUE]
        }
    }
    pub(super) fn operation(self) -> &'static str {
        if self.is_decode() {
            DECODE_OP
        } else if self.is_bf16() {
            BF16_OP
        } else if self.is_qk() {
            OP
        } else {
            VALUE_OP
        }
    }
    pub(super) fn output(self) -> &'static str {
        if self.is_decode() {
            DECODE_OUTPUT
        } else if self.is_bf16() {
            BF16_OUTPUT
        } else if self.is_qk() {
            OUTPUT
        } else {
            VALUE_OUTPUT
        }
    }
    pub(super) fn profile(self) -> &'static str {
        if self == Self::DecodeShift {
            DECODE_SHIFT_PROFILE
        } else if self == Self::DecodeNative {
            DECODE_NATIVE_PROFILE
        } else if self == Self::SteelReference {
            STEEL_PROFILE
        } else if self == Self::CompactDenseBf16 {
            COMPACT_DENSE_PROFILE
        } else if self == Self::ProductionBf16 {
            BF16_PROFILE
        } else if self == Self::ProductionNativeBf16 {
            NATIVE_BF16_PROFILE
        } else if self.is_qk() {
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
            Self::ProductionBf16 | Self::ProductionNativeBf16 => {
                "ferrum_tensor_attention_register128_bf16"
            }
            Self::SteelReference => steel::KERNEL,
            Self::CompactDenseBf16 => "ferrum_dense_attention_bf16_bq32_bk16_d128",
            Self::DecodeShift | Self::DecodeNative => "ferrum_diagnostic_bf16_decode",
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum ValueFixture {
    #[default]
    PositivePeriodic,
    ChannelConstant,
}
impl ValueFixture {
    fn is_positive(&self) -> bool {
        *self == Self::PositivePeriodic
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
    #[serde(default, skip_serializing_if = "ValueFixture::is_positive")]
    pub(super) value_fixture: ValueFixture,
}

impl Config {
    pub(super) fn validate(self) -> Result<Self, VNextError> {
        if self.batch == 0 || self.batch > 16 || self.heads == 0 {
            return Err(invalid(
                "attention diagnostic requires batch1..16 and nonzero heads",
            ));
        }
        if self.value_fixture == ValueFixture::ChannelConstant && !self.mode.is_bf16() {
            return Err(invalid(
                "constant-channel fixture requires a BF16 comparison mode",
            ));
        }
        if self.mode.uses_dense_params() {
            steel::Params::new(self)?;
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
        if self.mode.is_decode() && rows * columns > u32::MAX as u64 {
            return Err(invalid(
                "direct decode diagnostic exceeds u32 dispatch index",
            ));
        }
        rows.checked_mul(self.heads)
            .and_then(|n| n.checked_mul(self.mode.columns()))
            .and_then(|n| n.checked_mul(self.mode.output_element().size_bytes()))
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
            element_type: self.mode.output_element(),
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
                boundaries: BTreeMap::from([(
                    id(config.mode.output()),
                    config.mode.output_element(),
                )]),
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
            value_fixture: ValueFixture::PositivePeriodic,
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

#[test]
fn bf16_comparison_profiles_and_official_parameter_abi_are_explicit() {
    for mode in [
        Mode::ProductionBf16,
        Mode::ProductionNativeBf16,
        Mode::SteelReference,
        Mode::CompactDenseBf16,
    ] {
        let c = Config {
            batch: 2,
            sequence: 33,
            heads: 2,
            visibility: Visibility::AllValidBidirectional,
            mode,
            value_fixture: ValueFixture::PositivePeriodic,
        };
        TypedFamilyRegistration::new(Family)
            .prepare_with_profile(&serde_json::to_value(c).unwrap(), &id(mode.profile()))
            .unwrap();
        assert_eq!(c.output_spec().element_type, ElementType::Bf16);
        let p = steel::Params::new(c).unwrap();
        assert_eq!(p.q_strides, [8448, 128, 256]);
        assert_eq!(p.o_strides, p.q_strides);
        assert_eq!(
            (p.nq, p.nk, p.nq_aligned, p.nk_aligned, p.ql_rem, p.kl_rem),
            (2, 3, 1, 2, 1, 1)
        );
        assert_eq!(std::mem::size_of::<steel::Params>(), 152);
        assert_eq!(std::mem::align_of::<steel::Params>(), 8);
        assert_eq!(std::mem::offset_of!(steel::Params, q_strides), 56);
        assert!(steel::Params::new(Config {
            sequence: i32::MAX as u64 + 1,
            ..c
        })
        .is_err());
    }
}

#[test]
fn direct_decode_profiles_admit_one_bf16_input_and_real_f32_output() {
    for mode in [Mode::DecodeShift, Mode::DecodeNative] {
        let config = Config {
            batch: 1,
            sequence: 512,
            heads: 1,
            visibility: Visibility::AllValidBidirectional,
            mode,
            value_fixture: ValueFixture::PositivePeriodic,
        };
        TypedFamilyRegistration::new(Family)
            .prepare_with_profile(&serde_json::to_value(config).unwrap(), &id(mode.profile()))
            .unwrap();
        let descriptor = Contract::new(config);
        assert_eq!(descriptor.descriptor().inputs.len(), 1);
        assert_eq!(config.input_spec().element_type, ElementType::Bf16);
        assert_eq!(config.output_spec().element_type, ElementType::F32);
        assert_eq!(
            resolved(config.input_spec())
                .minimum_storage_bytes()
                .unwrap(),
            128 << 10
        );
        assert_eq!(
            resolved(config.output_spec())
                .minimum_storage_bytes()
                .unwrap(),
            256 << 10
        );
        assert!(
            Config {
                sequence: 1 << 25,
                ..config
            }
            .validate()
            .is_err(),
            "uint thread index must not wrap"
        );
    }
}
