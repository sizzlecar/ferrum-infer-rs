//! Exact checked video-query/full-key geometry for Sage's three FP4 layouts.
//! Query slicing and K centering are explicit producers outside this family.
use super::{FastH3VsaLayout, FAST_H3_VSA_TILE_ROWS};
use crate::fast_h3::FastH3TransformerConfig;
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug)]
pub struct FastH3VsaQkvQuantizeLimits {
    pub maximum_batches: u64,
    pub maximum_heads: u64,
    pub maximum_rows: u64,
    /// Sum of the three BF16 source extents, checked before compilation.
    pub maximum_input_bytes: u64,
    /// Sum of all six packed/scaled output extents.
    pub maximum_output_bytes: u64,
    pub maximum_metadata_bytes: usize,
}
impl FastH3VsaQkvQuantizeLimits {
    fn validate(self) -> Result<(), VNextError> {
        if [
            self.maximum_batches,
            self.maximum_heads,
            self.maximum_rows,
            self.maximum_input_bytes,
            self.maximum_output_bytes,
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
pub struct FastH3VsaQkvQuantizeConfig {
    batches: u64,
    heads: u64,
    query_rows: u64,
    key_rows: u64,
}
impl FastH3VsaQkvQuantizeConfig {
    pub fn from_layout(
        layout: &FastH3VsaLayout,
        batches: u64,
        transformer: &FastH3TransformerConfig,
        limits: FastH3VsaQkvQuantizeLimits,
    ) -> Result<Self, VNextError> {
        if transformer.attention_head_dim() != 128 {
            return Err(invalid("Sage profile requires original head dimension 128"));
        }
        let key_rows = u64::try_from(layout.padded_rows()).map_err(invalid)?;
        let prefix = u64::try_from(layout.prefix_tiles())
            .map_err(invalid)?
            .checked_mul(FAST_H3_VSA_TILE_ROWS as u64)
            .ok_or_else(|| invalid("prefix extent overflows"))?;
        let query_rows = key_rows
            .checked_sub(prefix)
            .ok_or_else(|| invalid("prefix exceeds keys"))?;
        let c = Self {
            batches,
            heads: transformer.attention_heads() as u64,
            query_rows,
            key_rows,
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
    pub fn query_rows(&self) -> u64 {
        self.query_rows
    }
    pub fn key_rows(&self) -> u64 {
        self.key_rows
    }
    pub fn input_shapes(&self) -> Result<[Vec<u64>; 3], VNextError> {
        let width = self
            .heads
            .checked_mul(128)
            .ok_or_else(|| invalid("head width overflows"))?;
        Ok([
            vec![self.batches, self.query_rows, width],
            vec![self.batches, self.key_rows, width],
            vec![self.batches, self.key_rows, width],
        ])
    }
    pub fn output_shapes(&self) -> [Vec<u64>; 6] {
        let (b, h, q, k) = (self.batches, self.heads, self.query_rows, self.key_rows);
        [
            vec![b, h, q, 64],
            vec![b, h, k, 64],
            vec![b, h, 128, k / 2],
            vec![b, h, q, 8],
            vec![b, h, k, 8],
            vec![b, h, 128, k / 16],
        ]
    }
    pub fn compile_options(
        &self,
        limits: FastH3VsaQkvQuantizeLimits,
    ) -> Result<ProgramPlanCompileOptions, VNextError> {
        self.validate(limits)?;
        ProgramPlanCompileOptions::new(self.specs()?)
    }
    fn specs(&self) -> Result<BTreeMap<ProgramValueId, ProgramTensorSpec>, VNextError> {
        let mut specs = self.inputs()?;
        for (name, shape) in FastH3VsaQkvQuantizeFamily::OUTPUT_IDS
            .into_iter()
            .zip(self.output_shapes())
        {
            specs.insert(id(name)?, tensor(shape, ElementType::U8));
        }
        Ok(specs)
    }
    fn inputs(&self) -> Result<BTreeMap<ProgramValueId, ProgramTensorSpec>, VNextError> {
        FastH3VsaQkvQuantizeFamily::INPUT_IDS
            .into_iter()
            .zip(self.input_shapes()?)
            .map(|(name, shape)| Ok((id(name)?, tensor(shape, ElementType::Bf16))))
            .collect()
    }
    fn validate(&self, limits: FastH3VsaQkvQuantizeLimits) -> Result<(), VNextError> {
        limits.validate()?;
        if self.batches > limits.maximum_batches
            || self.heads > limits.maximum_heads
            || self.query_rows >= self.key_rows
            || self.key_rows > limits.maximum_rows
        {
            return Err(invalid(
                "requires bounded batches/heads and a nonempty prefix before video queries",
            ));
        }
        let inputs = self.input_shapes()?.map(|s| resolved(s, ElementType::Bf16));
        let outputs = self.output_shapes().map(|s| resolved(s, ElementType::U8));
        let [q, k, v] = inputs;
        let [pq, pk, pv, sq, sk, sv] = outputs;
        let inputs = [q?, k?, v?];
        let outputs = [pq?, pk?, pv?, sq?, sk?, sv?];
        SparseFp4QkvQuantizeGeometry::new(
            inputs.each_ref(),
            outputs.each_ref(),
            &attributes(self.heads)?,
        )?;
        let bytes = |specs: &[ResolvedTensorSpec]| {
            specs.iter().try_fold(0u64, |n, s| {
                n.checked_add(s.minimum_storage_bytes()?)
                    .ok_or_else(|| invalid("tensor total overflows"))
            })
        };
        if bytes(&inputs)? > limits.maximum_input_bytes
            || bytes(&outputs)? > limits.maximum_output_bytes
            || serde_json::to_vec(self).map_err(invalid)?.len() > limits.maximum_metadata_bytes
        {
            return Err(invalid("input/output/metadata capacity exceeded"));
        }
        Ok(())
    }
}
pub struct FastH3VsaQkvQuantizeFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: FastH3VsaQkvQuantizeLimits,
}
impl FastH3VsaQkvQuantizeFamily {
    pub const FAMILY_ID: &'static str = "family.fast_h3.vsa.qkv_quantize";
    pub const PROFILE_ID: &'static str = "fast_h3.vsa.qkv_quantize.sage_group16_fast_math.v1";
    pub const QUERY_ID: &'static str = "value.fast_h3.vsa.qkv_quantize.query";
    pub const KEY_ID: &'static str = "value.fast_h3.vsa.qkv_quantize.centered_key";
    pub const VALUE_ID: &'static str = "value.fast_h3.vsa.qkv_quantize.value";
    pub const PACKED_QUERY_ID: &'static str = "value.fast_h3.vsa.qkv_quantize.packed_query";
    pub const PACKED_KEY_ID: &'static str = "value.fast_h3.vsa.qkv_quantize.packed_key";
    pub const PACKED_VALUE_ID: &'static str = "value.fast_h3.vsa.qkv_quantize.packed_value";
    pub const QUERY_SCALES_ID: &'static str = "value.fast_h3.vsa.qkv_quantize.query_scales";
    pub const KEY_SCALES_ID: &'static str = "value.fast_h3.vsa.qkv_quantize.key_scales";
    pub const VALUE_SCALES_ID: &'static str = "value.fast_h3.vsa.qkv_quantize.value_scales";
    pub const INPUT_IDS: [&'static str; 3] = [Self::QUERY_ID, Self::KEY_ID, Self::VALUE_ID];
    pub const OUTPUT_IDS: [&'static str; 6] = [
        Self::PACKED_QUERY_ID,
        Self::PACKED_KEY_ID,
        Self::PACKED_VALUE_ID,
        Self::QUERY_SCALES_ID,
        Self::KEY_SCALES_ID,
        Self::VALUE_SCALES_ID,
    ];
    pub fn new(limits: FastH3VsaQkvQuantizeLimits) -> Result<Self, VNextError> {
        limits.validate()?;
        Ok(Self {
            id: id(Self::FAMILY_ID)?,
            metadata: id("metadata.fast_h3.vsa.qkv_quantize")?,
            limits,
        })
    }
}
impl ModelFamilyProvider for FastH3VsaQkvQuantizeFamily {
    type Config = FastH3VsaQkvQuantizeConfig;
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
                primary_activation: id(Self::PACKED_QUERY_ID)?,
                boundaries: Self::OUTPUT_IDS
                    .into_iter()
                    .map(|name| Ok((id(name)?, ElementType::U8)))
                    .collect::<Result<_, VNextError>>()?,
                states: vec![],
                kv_storage: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(SPARSE_FP4_QKV_QUANTIZE_BF16_OPERATION_ID)?,
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
            return Err(invalid("wrong quantization profile"));
        }
        let inputs = Self::INPUT_IDS
            .into_iter()
            .map(id)
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = Self::OUTPUT_IDS
            .into_iter()
            .map(id)
            .collect::<Result<Vec<_>, _>>()?;
        ModelProgram::new(
            self.id.clone(),
            inputs.clone(),
            vec![ProgramBlock {
                id: "block.fast_h3.vsa.qkv_quantize".into(),
                nodes: vec![ProgramNode {
                    id: id("node.fast_h3.vsa.qkv_quantize")?,
                    operation_id: id(SPARSE_FP4_QKV_QUANTIZE_BF16_OPERATION_ID)?,
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs,
                    outputs: outputs.clone(),
                    attributes: attributes(c.heads)?,
                }],
            }],
            vec![],
            vec![],
            outputs,
        )?
        .with_input_tensor_specs(c.inputs()?)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn attributes(heads: u64) -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
    SparseFp4QkvQuantizeProfile::SageGroup16FastMathV1.attributes(heads)
}
fn tensor(dimensions: Vec<u64>, element_type: ElementType) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
fn resolved(d: Vec<u64>, ty: ElementType) -> Result<ResolvedTensorSpec, VNextError> {
    ResolvedTensorSpec::new(d, ty, ResolvedTensorLayout::Contiguous)
}
fn id<T: TryFrom<String, Error = VNextError>>(s: &str) -> Result<T, VNextError> {
    T::try_from(s.to_owned())
}
fn invalid(e: impl std::fmt::Display) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("FastH3 QKV quantization: {e}"),
    }
}

#[cfg(test)]
mod tests;
