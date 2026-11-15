//! Typed layout boundaries before checked VSA Untile and compression Expand.
use super::{FastH3VsaLayout, FastH3VsaPoolConfig};
use crate::fast_h3::FastH3TransformerConfig;
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug)]
pub struct FastH3HeadAxisPackLimits {
    pub maximum_batches: u64,
    pub maximum_rows: u64,
    pub maximum_columns: u64,
    pub maximum_input_bytes: u64,
    pub maximum_metadata_bytes: usize,
}
impl FastH3HeadAxisPackLimits {
    fn validate(self) -> Result<(), VNextError> {
        if [
            self.maximum_batches,
            self.maximum_rows,
            self.maximum_columns,
            self.maximum_input_bytes,
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
pub struct FastH3HeadAxisPackConfig {
    kind: HeadAxisPackKind,
    batches: u64,
    heads: u64,
    rows: u64,
    head_dim: u64,
}
impl FastH3HeadAxisPackConfig {
    /// Fine FP4 attention computes only the video-query suffix; the dense
    /// prefix is concatenated after this exact head permutation.
    pub fn from_video_layout(
        layout: &FastH3VsaLayout,
        batches: u64,
        transformer: &FastH3TransformerConfig,
        limits: FastH3HeadAxisPackLimits,
    ) -> Result<Self, VNextError> {
        let prefix = layout
            .prefix_tiles()
            .checked_mul(super::FAST_H3_VSA_TILE_ROWS)
            .ok_or_else(|| invalid("prefix extent overflow"))?;
        let rows = layout
            .padded_rows()
            .checked_sub(prefix)
            .filter(|&n| n > 0)
            .ok_or_else(|| invalid("video query suffix is empty"))?;
        Self::new(
            HeadAxisPackKind::Bf16,
            batches,
            transformer.attention_heads() as u64,
            rows.try_into().map_err(invalid)?,
            transformer.attention_head_dim() as u64,
            limits,
        )
    }
    pub fn from_fine_layout(
        layout: &FastH3VsaLayout,
        batches: u64,
        transformer: &FastH3TransformerConfig,
        limits: FastH3HeadAxisPackLimits,
    ) -> Result<Self, VNextError> {
        Self::new(
            HeadAxisPackKind::Bf16,
            batches,
            transformer.attention_heads() as u64,
            layout.padded_rows().try_into().map_err(invalid)?,
            transformer.attention_head_dim() as u64,
            limits,
        )
    }
    pub fn from_pool(
        pool: &FastH3VsaPoolConfig,
        limits: FastH3HeadAxisPackLimits,
    ) -> Result<Self, VNextError> {
        Self::new(
            HeadAxisPackKind::F32ToBf16,
            pool.batches(),
            pool.heads(),
            pool.tiles(),
            pool.head_dim(),
            limits,
        )
    }
    fn new(
        kind: HeadAxisPackKind,
        batches: u64,
        heads: u64,
        rows: u64,
        head_dim: u64,
        limits: FastH3HeadAxisPackLimits,
    ) -> Result<Self, VNextError> {
        let c = Self {
            kind,
            batches,
            heads,
            rows,
            head_dim,
        };
        c.validate(limits)?;
        Ok(c)
    }
    pub fn kind(&self) -> HeadAxisPackKind {
        self.kind
    }
    pub fn input_shape(&self) -> Vec<u64> {
        vec![self.batches, self.heads, self.rows, self.head_dim]
    }
    pub fn output_shape(&self) -> Result<Vec<u64>, VNextError> {
        Ok(vec![
            self.batches,
            self.rows,
            self.heads
                .checked_mul(self.head_dim)
                .ok_or_else(|| invalid("columns overflow"))?,
        ])
    }
    pub fn profile_id(&self) -> &'static str {
        match self.kind {
            HeadAxisPackKind::Bf16 => FastH3HeadAxisPackFamily::BF16_PROFILE_ID,
            HeadAxisPackKind::F32ToBf16 => FastH3HeadAxisPackFamily::F32_PROFILE_ID,
        }
    }
    pub fn compile_options(
        &self,
        limits: FastH3HeadAxisPackLimits,
    ) -> Result<ProgramPlanCompileOptions, VNextError> {
        self.validate(limits)?;
        ProgramPlanCompileOptions::new(self.specs()?)
    }
    fn specs(&self) -> Result<BTreeMap<ProgramValueId, ProgramTensorSpec>, VNextError> {
        Ok(BTreeMap::from([
            (
                id(FastH3HeadAxisPackFamily::INPUT_ID)?,
                tensor(self.input_shape(), self.kind.input_type()),
            ),
            (
                id(FastH3HeadAxisPackFamily::OUTPUT_ID)?,
                tensor(self.output_shape()?, ElementType::Bf16),
            ),
        ]))
    }
    fn validate(&self, limits: FastH3HeadAxisPackLimits) -> Result<(), VNextError> {
        limits.validate()?;
        let g = HeadAxisPackGeometry::new(
            self.kind,
            &ResolvedTensorSpec::new(
                self.input_shape(),
                self.kind.input_type(),
                ResolvedTensorLayout::Contiguous,
            )?,
            &ResolvedTensorSpec::new(
                self.output_shape()?,
                ElementType::Bf16,
                ResolvedTensorLayout::Contiguous,
            )?,
        )?;
        if self.batches > limits.maximum_batches
            || self.rows > limits.maximum_rows
            || self.output_shape()?[2] > limits.maximum_columns
            || g.input_bytes() > limits.maximum_input_bytes
            || serde_json::to_vec(self).map_err(invalid)?.len() > limits.maximum_metadata_bytes
        {
            return Err(invalid("geometry or metadata exceeds capacity"));
        }
        Ok(())
    }
}
pub struct FastH3HeadAxisPackFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: FastH3HeadAxisPackLimits,
}
impl FastH3HeadAxisPackFamily {
    pub const FAMILY_ID: &'static str = "family.fast_h3.vsa.head_axis_pack";
    pub const BF16_PROFILE_ID: &'static str = "fast_h3.vsa.head_axis_pack.bf16_bits.v1";
    pub const F32_PROFILE_ID: &'static str = "fast_h3.vsa.head_axis_pack.f32_bf16_rne.v1";
    pub const INPUT_ID: &'static str = "value.fast_h3.vsa.head_axis_pack.input";
    pub const OUTPUT_ID: &'static str = "value.fast_h3.vsa.head_axis_pack.output";
    pub fn new(limits: FastH3HeadAxisPackLimits) -> Result<Self, VNextError> {
        limits.validate()?;
        Ok(Self {
            id: id(Self::FAMILY_ID)?,
            metadata: id("metadata.fast_h3.vsa.head_axis_pack")?,
            limits,
        })
    }
}
impl ModelFamilyProvider for FastH3HeadAxisPackFamily {
    type Config = FastH3HeadAxisPackConfig;
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
                id: NumericalProfileId::new(c.profile_id()).map_err(invalid)?,
                family_id: self.id.clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: id(Self::OUTPUT_ID)?,
                boundaries: BTreeMap::from([(id(Self::OUTPUT_ID)?, ElementType::Bf16)]),
                states: vec![],
                kv_storage: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(c.kind.operation_id())?,
                    version: ContractVersion::new(1, 0),
                    multiplication_type: None,
                    accumulation_type: None,
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                }],
            }],
            vec![NumericalProfileId::new(c.profile_id()).map_err(invalid)?],
        )
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        if self.numerical_profiles(c)?.resolve(&p.id)? != p {
            return Err(invalid("wrong layout/conversion profile"));
        }
        ModelProgram::new(
            self.id.clone(),
            vec![id(Self::INPUT_ID)?],
            vec![ProgramBlock {
                id: "block.fast_h3.vsa.head_axis_pack".into(),
                nodes: vec![ProgramNode {
                    id: id("node.fast_h3.vsa.head_axis_pack")?,
                    operation_id: id(c.kind.operation_id())?,
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: vec![id(Self::INPUT_ID)?],
                    outputs: vec![id(Self::OUTPUT_ID)?],
                    attributes: BTreeMap::new(),
                }],
            }],
            vec![],
            vec![],
            vec![id(Self::OUTPUT_ID)?],
        )?
        .with_input_tensor_specs(BTreeMap::from([(
            id(Self::INPUT_ID)?,
            tensor(c.input_shape(), c.kind.input_type()),
        )]))
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn tensor(dimensions: Vec<u64>, element_type: ElementType) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
fn id<T: TryFrom<String, Error = VNextError>>(s: &str) -> Result<T, VNextError> {
    T::try_from(s.to_owned())
}
fn invalid(e: impl std::fmt::Display) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("FastH3 head-axis pack: {e}"),
    }
}

#[cfg(test)]
mod tests;
