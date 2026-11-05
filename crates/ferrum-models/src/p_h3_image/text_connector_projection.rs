//! Executable bias-free F32 projection of the P-H3 text connector. CPU Host
//! adapters own tap normalization and final sink replacement; this stored
//! TensorComponent runs the actual heavy matrix product via the ordinary
//! F32 dense operation. It is not a complete image pipeline.

use std::collections::{BTreeMap, BTreeSet};

use ferrum_runtime::*;
use serde::{Deserialize, Deserializer, Serialize};

use super::PH3ImageTextConnectorConfig;

/// Explicit source/metadata capacity, not an implicit GPU admission grant.
#[derive(Clone, Copy, Debug)]
pub struct PH3ImageTextConnectorProjectionLimits {
    pub max_projection_bytes: u64,
}

/// Exact checkpoint schemas. Comfy stores the six F32 connector parameters
/// alongside the quantized encoder; no prefix guessing or dtype conversion is
/// involved in selecting those original tensors.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PH3ImageTextConnectorCheckpointLayout {
    #[default]
    DiffusersSeparate,
    ComfyMergedTextEncoder,
}
pub(super) struct CheckpointNames {
    pub projection: &'static str,
    pub input_mean: &'static str,
    pub input_std: &'static str,
    pub output_mean: &'static str,
    pub output_std: &'static str,
    pub sink: &'static str,
}
impl PH3ImageTextConnectorCheckpointLayout {
    fn is_default(&self) -> bool {
        *self == Self::DiffusersSeparate
    }
    pub(super) fn names(self) -> CheckpointNames {
        match self {
            Self::DiffusersSeparate => CheckpointNames {
                projection: "proj.weight",
                input_mean: "input_mean",
                input_std: "input_std",
                output_mean: "output_mean",
                output_std: "output_std",
                sink: "sink_embedding",
            },
            Self::ComfyMergedTextEncoder => CheckpointNames {
                projection: "text_connector.proj.weight",
                input_mean: "text_connector.input_mean",
                input_std: "text_connector.input_std",
                output_mean: "text_connector.output_mean",
                output_std: "text_connector.output_std",
                sink: "text_connector.sink_embedding",
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PH3ImageTextConnectorProjectionConfig {
    input_dim: u64,
    output_dim: u64,
    #[serde(
        default,
        skip_serializing_if = "PH3ImageTextConnectorCheckpointLayout::is_default"
    )]
    checkpoint_layout: PH3ImageTextConnectorCheckpointLayout,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigWire {
    input_dim: u64,
    output_dim: u64,
    #[serde(default)]
    checkpoint_layout: PH3ImageTextConnectorCheckpointLayout,
}

impl<'de> Deserialize<'de> for PH3ImageTextConnectorProjectionConfig {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = ConfigWire::deserialize(deserializer)?;
        Self::new(raw.input_dim, raw.output_dim)
            .map(|config| config.with_checkpoint_layout(raw.checkpoint_layout))
            .map_err(serde::de::Error::custom)
    }
}
impl PH3ImageTextConnectorProjectionConfig {
    pub fn new(input_dim: u64, output_dim: u64) -> Result<Self, VNextError> {
        let config = Self {
            input_dim,
            output_dim,
            checkpoint_layout: PH3ImageTextConnectorCheckpointLayout::default(),
        };
        config.weight_spec().validate("connector projection")?;
        let bytes = config.projection_bytes()?;
        if input_dim == 0 || output_dim == 0 || bytes > isize::MAX as u64 {
            return Err(invalid(
                "invalid connector projection dimensions or address-space extent",
            ));
        }
        Ok(config)
    }
    pub fn from_connector(config: &PH3ImageTextConnectorConfig) -> Result<Self, VNextError> {
        Self::new(config.input_dim(), config.out_channels())
    }
    pub fn input_dim(&self) -> u64 {
        self.input_dim
    }
    pub fn output_dim(&self) -> u64 {
        self.output_dim
    }
    pub fn checkpoint_layout(&self) -> PH3ImageTextConnectorCheckpointLayout {
        self.checkpoint_layout
    }
    pub fn with_checkpoint_layout(mut self, layout: PH3ImageTextConnectorCheckpointLayout) -> Self {
        self.checkpoint_layout = layout;
        self
    }
    pub fn projection_bytes(&self) -> Result<u64, VNextError> {
        self.input_dim
            .checked_mul(self.output_dim)
            .and_then(|count| count.checked_mul(4))
            .ok_or_else(|| invalid("connector projection payload overflows"))
    }
    pub fn input_spec(&self, rows: u64) -> Result<ProgramTensorSpec, VNextError> {
        let spec = ProgramTensorSpec {
            dimensions: vec![rows, self.input_dim],
            element_type: ElementType::F32,
            layout: ResolvedTensorLayout::Contiguous,
        };
        spec.validate("connector normalized input")?;
        if spec.byte_len()? > isize::MAX as u64 {
            return Err(invalid("connector input exceeds address space"));
        }
        Ok(spec)
    }
    fn weight_spec(&self) -> ProgramTensorSpec {
        ProgramTensorSpec {
            dimensions: vec![self.output_dim, self.input_dim],
            element_type: ElementType::F32,
            layout: ResolvedTensorLayout::Contiguous,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PH3ImageTextConnectorProjectionFamily {
    family_id: ModelFamilyId,
    metadata_id: ExternalModelMetadataId,
    limits: PH3ImageTextConnectorProjectionLimits,
}
impl PH3ImageTextConnectorProjectionFamily {
    pub const FAMILY_ID: &'static str = "family.p_h3_image.text_connector.projection";
    pub const REFERENCE_PROFILE_ID: &'static str =
        "p_h3_image.text_connector.projection.reference.f32.v1";
    pub const CUDA_PROFILE_ID: &'static str = "p_h3_image.text_connector.projection.cuda.f32.v1";
    pub const INPUT_ID: &'static str = "value.p_h3_image.text_connector.normalized";
    pub const OUTPUT_ID: &'static str = "value.p_h3_image.text_connector.projected";
    pub const WEIGHT_ID: &'static str = "weight.p_h3_image.text_connector.proj.weight";
    pub const COMPONENT_ID: &'static str = "weight.component.p_h3_image.text_connector.proj.weight";
    pub const EXTERNAL_WEIGHT_NAME: &'static str = "proj.weight";
    const WEIGHT_VALUE_ID: &'static str = "value.p_h3_image.text_connector.proj.weight";
    pub fn new(limits: PH3ImageTextConnectorProjectionLimits) -> Result<Self, VNextError> {
        if limits.max_projection_bytes == 0 {
            return Err(invalid("connector projection capacity must be positive"));
        }
        Ok(Self {
            family_id: ModelFamilyId::new(Self::FAMILY_ID)?,
            metadata_id: ExternalModelMetadataId::new(
                "metadata.p_h3_image.text_connector.projection",
            )?,
            limits,
        })
    }
    fn validate_capacity(
        &self,
        config: &PH3ImageTextConnectorProjectionConfig,
    ) -> Result<(), VNextError> {
        if config.projection_bytes()? > self.limits.max_projection_bytes {
            return Err(invalid(
                "connector projection exceeds declared source capacity",
            ));
        }
        Ok(())
    }
}
impl ModelFamilyProvider for PH3ImageTextConnectorProjectionFamily {
    type Config = PH3ImageTextConnectorProjectionConfig;
    fn family_id(&self) -> &ModelFamilyId {
        &self.family_id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([self.metadata_id.clone()])
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        config: &Self::Config,
    ) -> Result<(), VNextError> {
        if serde_json::to_value(config).map_err(|error| invalid(error.to_string()))? != *raw {
            return Err(invalid(
                "connector projection configuration must be canonical",
            ));
        }
        self.validate_capacity(config)
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        config: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(self.metadata_id.clone())
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Self::Config, VNextError> {
        let config =
            serde_json::from_value(raw.clone()).map_err(|error| invalid(error.to_string()))?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }
    fn weights(&self, config: &Self::Config) -> Result<ModelWeights, VNextError> {
        self.validate_capacity(config)?;
        let component = WeightId::new(Self::COMPONENT_ID)?;
        let schema = WeightSchema {
            format_id: WeightFormatId::new("weight-format.safetensors.dense")?,
            layout_id: WeightLayoutId::new(
                "weight-layout.p_h3_image.text_connector.projection.f32",
            )?,
            version: ContractVersion::new(1, 0),
            components: vec![WeightComponentSpec {
                id: component.clone(),
                role: WeightComponentRole::Values,
                external_names: vec![config.checkpoint_layout.names().projection.into()],
                dimensions: vec![config.output_dim, config.input_dim],
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::F32,
                },
                required: true,
            }],
            tensors: vec![WeightTensorSpec {
                id: WeightId::new(Self::WEIGHT_ID)?,
                dimensions: vec![config.output_dim, config.input_dim],
                logical_element_type: ElementType::F32,
                physical_layout: PhysicalWeightLayout::Dense {
                    component_id: component,
                },
                required: true,
            }],
        };
        schema.validate(&self.family_id)?;
        Ok(ModelWeights::Stored(schema))
    }
    fn numerical_profiles(
        &self,
        config: &Self::Config,
    ) -> Result<FamilyNumericalProfiles, VNextError> {
        self.validate_capacity(config)?;
        let profiles = [Self::REFERENCE_PROFILE_ID, Self::CUDA_PROFILE_ID]
            .into_iter()
            .map(|name| {
                Ok(NumericalExecutionProfile {
                    id: NumericalProfileId::new(name).map_err(invalid)?,
                    family_id: self.family_id.clone(),
                    version: ContractVersion::new(1, 0),
                    primary_activation: ProgramValueId::new(Self::OUTPUT_ID)?,
                    boundaries: BTreeMap::from([(
                        ProgramValueId::new(Self::OUTPUT_ID)?,
                        ElementType::F32,
                    )]),
                    states: vec![],
                    kv_storage: vec![],
                    operations: vec![NumericalOperationContract {
                        operation_id: OperationId::new(DENSE_LINEAR_F32_OPERATION_ID)?,
                        version: ContractVersion::new(1, 0),
                        multiplication_type: Some(ElementType::F32),
                        accumulation_type: Some(ElementType::F32),
                        staged_arithmetic: None,
                        composite_arithmetic: None,
                    }],
                })
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
        // Profiles declare arithmetic boundaries. Real provider/library identity,
        // error qualification and CUDA reduction evidence belong to execution.
        FamilyNumericalProfiles::new(
            &self.family_id,
            ContractVersion::new(1, 0),
            profiles,
            vec![NumericalProfileId::new(Self::REFERENCE_PROFILE_ID).map_err(invalid)?],
        )
    }
    fn semantic_program(
        &self,
        config: &Self::Config,
        profile: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        if self.numerical_profiles(config)?.resolve(&profile.id)? != profile {
            return Err(invalid("unknown connector projection numerical profile"));
        }
        let input = ProgramValueId::new(Self::INPUT_ID)?;
        let weight = ProgramValueId::new(Self::WEIGHT_VALUE_ID)?;
        let output = ProgramValueId::new(Self::OUTPUT_ID)?;
        ModelProgram::new(
            self.family_id.clone(),
            vec![input.clone()],
            vec![ProgramBlock {
                id: "block.p_h3_image.text_connector.projection".into(),
                nodes: vec![ProgramNode {
                    id: NodeId::new("node.p_h3_image.text_connector.projection")?,
                    operation_id: OperationId::new(DENSE_LINEAR_F32_OPERATION_ID)?,
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: vec![input, weight.clone()],
                    outputs: vec![output.clone()],
                    attributes: BTreeMap::from([
                        (
                            AttributeId::new("in_features")?,
                            SemanticValue::Unsigned(config.input_dim),
                        ),
                        (
                            AttributeId::new("out_features")?,
                            SemanticValue::Unsigned(config.output_dim),
                        ),
                    ]),
                }],
            }],
            vec![],
            vec![WeightReference {
                weight_id: WeightId::new(Self::WEIGHT_ID)?,
                value_id: weight,
                tensor: config.weight_spec(),
            }],
            vec![output],
        )
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}
