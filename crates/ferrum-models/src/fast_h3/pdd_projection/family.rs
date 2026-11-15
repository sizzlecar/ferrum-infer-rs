//! One F32 affine operation over explicitly prepared PDD parameters.
use super::{invalid, program, FastH3PddProjectionConfig, FastH3PddProjectionLimits};
use ferrum_runtime::*;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub struct FastH3PddProjectionFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: FastH3PddProjectionLimits,
}
impl FastH3PddProjectionFamily {
    pub const FAMILY_ID: &'static str = "family.fast_h3.pdd_projection";
    pub const PROFILE_ID: &'static str = "fast_h3.pdd_projection.f32_dot_then_bias.v1";
    pub const INPUT_ID: &'static str = "value.fast_h3.pdd_projection.normalized";
    pub const OUTPUT_ID: &'static str = "value.fast_h3.pdd_projection.velocity";

    pub fn new(limits: FastH3PddProjectionLimits) -> Result<Self, VNextError> {
        if limits.maximum_rows == 0
            || limits.maximum_metadata_bytes == 0
            || limits.maximum_source_bytes == 0
            || limits.maximum_fused_bytes == 0
        {
            return Err(invalid("PDD projection capacities must be positive"));
        }
        Ok(Self {
            id: ModelFamilyId::new(Self::FAMILY_ID)?,
            metadata: ExternalModelMetadataId::new("metadata.fast_h3.pdd_projection")?,
            limits,
        })
    }
}

pub(super) fn weight(index: usize) -> Result<WeightId, VNextError> {
    WeightId::new(format!("weight.fast_h3.pdd_projection.{index}"))
}
pub(super) fn schema(c: &FastH3PddProjectionConfig) -> Result<WeightSchema, VNextError> {
    let schema = WeightSchema {
        format_id: WeightFormatId::new("weight-format.safetensors.dense")?,
        layout_id: WeightLayoutId::new("weight-layout.fast_h3.pdd_projection.fused_f32.v1")?,
        version: ContractVersion::new(1, 0),
        components: c.parameters.iter().map(|p| p.fused.clone()).collect(),
        tensors: c
            .parameters
            .iter()
            .enumerate()
            .map(|(i, p)| {
                Ok(WeightTensorSpec {
                    id: weight(i)?,
                    dimensions: p.fused.dimensions.clone(),
                    logical_element_type: ElementType::F32,
                    physical_layout: PhysicalWeightLayout::Dense {
                        component_id: p.fused.id.clone(),
                    },
                    required: true,
                })
            })
            .collect::<Result<_, VNextError>>()?,
    };
    schema.validate(&ModelFamilyId::new(FastH3PddProjectionFamily::FAMILY_ID)?)?;
    Ok(schema)
}

impl ModelFamilyProvider for FastH3PddProjectionFamily {
    type Config = FastH3PddProjectionConfig;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([self.metadata.clone()])
    }
    fn validate_config_identity(&self, raw: &Value, c: &Self::Config) -> Result<(), VNextError> {
        if serde_json::to_value(c).map_err(|e| invalid(e.to_string()))? != *raw {
            return Err(invalid("PDD projection configuration is not canonical"));
        }
        c.validate(self.limits)
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        c: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(self.metadata.clone())
    }
    fn parse_config(&self, raw: &Value) -> Result<Self::Config, VNextError> {
        let c = serde_json::from_value(raw.clone()).map_err(|e| invalid(e.to_string()))?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn weights(&self, c: &Self::Config) -> Result<ModelWeights, VNextError> {
        c.validate(self.limits)?;
        Ok(ModelWeights::Stored(schema(c)?))
    }
    fn numerical_profiles(&self, c: &Self::Config) -> Result<FamilyNumericalProfiles, VNextError> {
        c.validate(self.limits)?;
        let id = NumericalProfileId::new(Self::PROFILE_ID).map_err(invalid)?;
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: id.clone(),
                family_id: self.id.clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: ProgramValueId::new(Self::OUTPUT_ID)?,
                boundaries: BTreeMap::from([(
                    ProgramValueId::new(Self::OUTPUT_ID)?,
                    ElementType::F32,
                )]),
                states: vec![],
                kv_storage: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: OperationId::new(AFFINE_LINEAR_F32_OPERATION_ID)?,
                    version: ContractVersion::new(1, 0),
                    multiplication_type: Some(ElementType::F32),
                    accumulation_type: Some(ElementType::F32),
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                }],
            }],
            vec![id],
        )
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        profile: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        if self.numerical_profiles(c)?.resolve(&profile.id)? != profile {
            return Err(invalid("unknown PDD projection numerical profile"));
        }
        Ok(program::build(c)?.0)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
