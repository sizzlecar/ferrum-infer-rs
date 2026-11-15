//! Key smoothing after pooling of the original, unsmoothed Q/K/V.
//! The checked layout's entire padded sequence participates in the mean.
use super::FastH3VsaLayout;
use crate::fast_h3::FastH3TransformerConfig;
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug)]
pub struct FastH3KeyCenterLimits {
    pub maximum_batches: u64,
    pub maximum_rows: u64,
    pub maximum_columns: u64,
    pub maximum_tensor_bytes: u64,
    pub maximum_metadata_bytes: usize,
}
impl FastH3KeyCenterLimits {
    fn validate(self) -> Result<(), VNextError> {
        if [
            self.maximum_batches,
            self.maximum_rows,
            self.maximum_columns,
            self.maximum_tensor_bytes,
        ]
        .contains(&0)
            || self.maximum_metadata_bytes == 0
        {
            return Err(invalid("capacities must be positive"));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastH3KeyCenterConfig {
    batches: u64,
    padded_rows: u64,
    columns: u64,
}
impl FastH3KeyCenterConfig {
    pub fn from_layout(
        layout: &FastH3VsaLayout,
        batches: u64,
        transformer: &FastH3TransformerConfig,
        limits: FastH3KeyCenterLimits,
    ) -> Result<Self, VNextError> {
        let columns = (transformer.attention_heads() as u64)
            .checked_mul(transformer.attention_head_dim() as u64)
            .ok_or_else(|| invalid("head width overflows"))?;
        let c = Self {
            batches,
            padded_rows: layout.padded_rows().try_into().map_err(invalid)?,
            columns,
        };
        c.validate(limits)?;
        Ok(c)
    }
    pub fn input_shape(&self) -> Vec<u64> {
        vec![self.batches, self.padded_rows, self.columns]
    }
    pub fn mean_shape(&self) -> Vec<u64> {
        vec![self.batches, 1, self.columns]
    }
    pub fn compile_options(
        &self,
        limits: FastH3KeyCenterLimits,
    ) -> Result<ProgramPlanCompileOptions, VNextError> {
        self.validate(limits)?;
        ProgramPlanCompileOptions::new(BTreeMap::from([
            (
                id(FastH3KeyCenterFamily::INPUT_ID)?,
                tensor(self.input_shape()),
            ),
            (
                id(FastH3KeyCenterFamily::OUTPUT_ID)?,
                tensor(self.input_shape()),
            ),
            (
                id(FastH3KeyCenterFamily::MEAN_ID)?,
                tensor(self.mean_shape()),
            ),
        ]))
    }
    fn validate(&self, limits: FastH3KeyCenterLimits) -> Result<(), VNextError> {
        limits.validate()?;
        let resolved =
            |d| ResolvedTensorSpec::new(d, ElementType::Bf16, ResolvedTensorLayout::Contiguous);
        let x = resolved(self.input_shape())?;
        let g = RowCenterBf16Geometry::new(
            &x,
            &x,
            &resolved(self.mean_shape())?,
            &RowCenterBf16Geometry::attributes()?,
        )?;
        if self.batches > limits.maximum_batches
            || self.padded_rows > limits.maximum_rows
            || self.columns > limits.maximum_columns
            || g.data_bytes() > limits.maximum_tensor_bytes
            || serde_json::to_vec(self).map_err(invalid)?.len() > limits.maximum_metadata_bytes
        {
            return Err(invalid("geometry or metadata exceeds capacity"));
        }
        Ok(())
    }
}
pub struct FastH3KeyCenterFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: FastH3KeyCenterLimits,
}
impl FastH3KeyCenterFamily {
    pub const FAMILY_ID: &'static str = "family.fast_h3.vsa.key_center";
    pub const PROFILE_ID: &'static str =
        "fast_h3.vsa.key_center.f32_stride8_tree_mean_bf16_subtract_rn.v1";
    pub const INPUT_ID: &'static str = "value.fast_h3.vsa.key_center.input";
    pub const OUTPUT_ID: &'static str = "value.fast_h3.vsa.key_center.output";
    pub const MEAN_ID: &'static str = "value.fast_h3.vsa.key_center.mean";
    pub fn new(limits: FastH3KeyCenterLimits) -> Result<Self, VNextError> {
        limits.validate()?;
        Ok(Self {
            id: id(Self::FAMILY_ID)?,
            metadata: id("metadata.fast_h3.vsa.key_center")?,
            limits,
        })
    }
}
impl ModelFamilyProvider for FastH3KeyCenterFamily {
    type Config = FastH3KeyCenterConfig;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([self.metadata.clone()])
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<(), VNextError> {
        if serde_json::to_value(c).map_err(invalid)? != *raw {
            return Err(invalid("noncanonical config"));
        }
        c.validate(self.limits)
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(self.metadata.clone())
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Self::Config, VNextError> {
        let c = serde_json::from_value(raw.clone()).map_err(invalid)?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn weights(&self, c: &Self::Config) -> Result<ModelWeights, VNextError> {
        c.validate(self.limits)?;
        Ok(ModelWeights::None)
    }
    fn numerical_profiles(&self, c: &Self::Config) -> Result<FamilyNumericalProfiles, VNextError> {
        c.validate(self.limits)?;
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: NumericalProfileId::new(Self::PROFILE_ID).map_err(invalid)?,
                family_id: self.id.clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: id(Self::OUTPUT_ID)?,
                boundaries: BTreeMap::from([
                    (id(Self::OUTPUT_ID)?, ElementType::Bf16),
                    (id(Self::MEAN_ID)?, ElementType::Bf16),
                ]),
                states: vec![],
                kv_storage: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(ROW_CENTER_BF16_OPERATION_ID)?,
                    version: ContractVersion::new(1, 0),
                    multiplication_type: Some(ElementType::F32),
                    accumulation_type: Some(ElementType::F32),
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                }],
            }],
            vec![NumericalProfileId::new(Self::PROFILE_ID).map_err(invalid)?],
        )
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        if self.numerical_profiles(c)?.resolve(&p.id)? != p {
            return Err(invalid("wrong numerical profile"));
        }
        ModelProgram::new(
            self.id.clone(),
            vec![id(Self::INPUT_ID)?],
            vec![ProgramBlock {
                id: "block.fast_h3.vsa.key_center".into(),
                nodes: vec![ProgramNode {
                    id: id("node.fast_h3.vsa.key_center")?,
                    operation_id: id(ROW_CENTER_BF16_OPERATION_ID)?,
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: vec![id(Self::INPUT_ID)?],
                    outputs: vec![id(Self::OUTPUT_ID)?, id(Self::MEAN_ID)?],
                    attributes: RowCenterBf16Geometry::attributes()?,
                }],
            }],
            vec![],
            vec![],
            vec![id(Self::OUTPUT_ID)?, id(Self::MEAN_ID)?],
        )?
        .with_input_tensor_specs(BTreeMap::from([(
            id(Self::INPUT_ID)?,
            tensor(c.input_shape()),
        )]))
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn tensor(dimensions: Vec<u64>) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type: ElementType::Bf16,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
fn id<T: TryFrom<String, Error = VNextError>>(s: &str) -> Result<T, VNextError> {
    T::try_from(s.to_owned())
}
fn invalid(e: impl std::fmt::Display) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("FastH3 key centering: {e}"),
    }
}
#[cfg(test)]
mod tests;
