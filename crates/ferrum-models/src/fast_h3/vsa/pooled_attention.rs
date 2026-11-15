//! Shared scaled scores and dense compression over checked tile-pool geometry.
//! Q/K/V are the actual F32 [B,H,T,D] pool outputs. Scores are materialized once
//! and returned for sparse selection while the same value feeds softmax/PV.
use super::FastH3VsaPoolConfig;
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
#[cfg(all(test, feature = "cuda"))]
mod cuda;
mod program;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug)]
pub struct FastH3PooledAttentionLimits {
    pub maximum_batches: u64,
    pub maximum_heads: u64,
    pub maximum_tiles: u64,
    pub maximum_head_dim: u64,
    pub maximum_score_bytes: u64,
    pub maximum_metadata_bytes: usize,
}
impl FastH3PooledAttentionLimits {
    fn validate(self) -> Result<(), VNextError> {
        if [
            self.maximum_batches,
            self.maximum_heads,
            self.maximum_tiles,
            self.maximum_head_dim,
            self.maximum_score_bytes,
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
pub struct FastH3PooledAttentionConfig {
    batches: u64,
    heads: u64,
    tiles: u64,
    head_dim: u64,
    scale_bits: u32,
}
impl FastH3PooledAttentionConfig {
    pub fn from_pool(
        pool: &FastH3VsaPoolConfig,
        limits: FastH3PooledAttentionLimits,
    ) -> Result<Self, VNextError> {
        Self::new(
            pool.batches(),
            pool.heads(),
            pool.tiles(),
            pool.head_dim(),
            limits,
        )
    }
    fn new(
        batches: u64,
        heads: u64,
        tiles: u64,
        head_dim: u64,
        limits: FastH3PooledAttentionLimits,
    ) -> Result<Self, VNextError> {
        let c = Self {
            batches,
            heads,
            tiles,
            head_dim,
            scale_bits: scale(head_dim)?.to_bits(),
        };
        c.validate(limits)?;
        Ok(c)
    }
    pub fn batches(&self) -> u64 {
        self.batches
    }
    pub fn heads(&self) -> u64 {
        self.heads
    }
    pub fn tiles(&self) -> u64 {
        self.tiles
    }
    pub fn head_dim(&self) -> u64 {
        self.head_dim
    }
    pub fn scale_bits(&self) -> u32 {
        self.scale_bits
    }
    pub fn pooled_shape(&self) -> Vec<u64> {
        vec![self.batches, self.heads, self.tiles, self.head_dim]
    }
    pub fn score_shape(&self) -> Vec<u64> {
        vec![self.batches, self.heads, self.tiles, self.tiles]
    }
    pub fn compile_options(
        &self,
        limits: FastH3PooledAttentionLimits,
    ) -> Result<ProgramPlanCompileOptions, VNextError> {
        self.validate(limits)?;
        ProgramPlanCompileOptions::new(program::build(self)?.1)
    }
    fn validate(&self, limits: FastH3PooledAttentionLimits) -> Result<(), VNextError> {
        limits.validate()?;
        if self.batches == 0
            || self.batches > limits.maximum_batches
            || self.heads == 0
            || self.heads > limits.maximum_heads
            || self.tiles == 0
            || self.tiles > limits.maximum_tiles
            || self.head_dim == 0
            || self.head_dim > limits.maximum_head_dim
            || self.scale_bits != scale(self.head_dim)?.to_bits()
        {
            return Err(invalid("invalid geometry, scale derivation or capacity"));
        }
        let pooled = resolved(self.pooled_shape())?;
        let scores = resolved(self.score_shape())?;
        if scores.minimum_storage_bytes()? > limits.maximum_score_bytes
            || serde_json::to_vec(self).map_err(invalid)?.len() > limits.maximum_metadata_bytes
        {
            return Err(invalid("score or metadata capacity exceeded"));
        }
        BatchedF32Geometry::new(
            BatchedF32Op::Matmul,
            &[&pooled, &pooled],
            &scores,
            &BatchedF32Geometry::matmul_attributes(true)?,
        )?;
        BatchedF32Geometry::new(
            BatchedF32Op::Matmul,
            &[&scores, &pooled],
            &pooled,
            &BatchedF32Geometry::matmul_attributes(false)?,
        )?;
        Ok(())
    }
}
/// Fixed FastVideo `scores / (dim**0.5)` with a CPU double divisor. Pinned
/// Torch BinaryDivTrueKernel computes a double reciprocal, casts it to F32,
/// then multiplies on-device. The cast is before, not inside, the GEMM.
fn scale(head_dim: u64) -> Result<f32, VNextError> {
    if head_dim == 0 || head_dim > (1 << 53) {
        return Err(invalid(
            "head dimension is not exact in the source F64 scalar domain",
        ));
    }
    Ok((1.0f64 / (head_dim as f64).sqrt()) as f32)
}
pub struct FastH3PooledAttentionFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: FastH3PooledAttentionLimits,
}
impl FastH3PooledAttentionFamily {
    pub const FAMILY_ID: &'static str = "family.fast_h3.vsa.pooled_attention";
    pub const PROFILE_ID: &'static str = "fast_h3.vsa.pooled_attention.f32_shared_scaled_scores.v1";
    pub const QUERY_ID: &'static str = "value.fast_h3.vsa.pooled_attention.query";
    pub const KEY_ID: &'static str = "value.fast_h3.vsa.pooled_attention.key";
    pub const VALUE_ID: &'static str = "value.fast_h3.vsa.pooled_attention.value";
    pub const SCORES_ID: &'static str = "value.fast_h3.vsa.pooled_attention.scaled_scores";
    pub const OUTPUT_ID: &'static str = "value.fast_h3.vsa.pooled_attention.weighted_value";
    pub fn new(limits: FastH3PooledAttentionLimits) -> Result<Self, VNextError> {
        limits.validate()?;
        Ok(Self {
            id: id(Self::FAMILY_ID)?,
            metadata: id("metadata.fast_h3.vsa.pooled_attention")?,
            limits,
        })
    }
}
impl ModelFamilyProvider for FastH3PooledAttentionFamily {
    type Config = FastH3PooledAttentionConfig;
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
            return Err(invalid("noncanonical configuration"));
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
        let (program, _) = program::build(c)?;
        let boundaries = program
            .blocks()
            .iter()
            .flat_map(|b| b.nodes.iter())
            .flat_map(|n| n.outputs.iter())
            .map(|v| (v.clone(), ElementType::F32))
            .collect();
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: NumericalProfileId::new(Self::PROFILE_ID).map_err(invalid)?,
                family_id: self.id.clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: id(Self::OUTPUT_ID)?,
                boundaries,
                states: vec![],
                kv_storage: vec![],
                operations: BatchedF32Op::ALL
                    .into_iter()
                    .map(|op| {
                        Ok(NumericalOperationContract {
                            operation_id: id(op.operation_id())?,
                            version: ContractVersion::new(1, 0),
                            multiplication_type: Some(ElementType::F32),
                            accumulation_type: Some(ElementType::F32),
                            staged_arithmetic: None,
                            composite_arithmetic: None,
                        })
                    })
                    .collect::<Result<_, VNextError>>()?,
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
            return Err(invalid("unknown profile"));
        }
        Ok(program::build(c)?.0)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn id<T: TryFrom<String, Error = VNextError>>(s: &str) -> Result<T, VNextError> {
    T::try_from(s.to_owned())
}
fn resolved(shape: Vec<u64>) -> Result<ResolvedTensorSpec, VNextError> {
    ResolvedTensorSpec::new(shape, ElementType::F32, ResolvedTensorLayout::Contiguous)
}
fn tensor(dimensions: Vec<u64>) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type: ElementType::F32,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
fn invalid(e: impl std::fmt::Display) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("FastH3 pooled attention: {e}"),
    }
}
