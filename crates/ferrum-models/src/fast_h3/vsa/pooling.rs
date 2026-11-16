//! Positive checked VSA prefix counts, stored once for Q/K/V tile pooling.
use super::{FastH3VsaLayout, FAST_H3_VSA_TILE_ROWS};
use crate::fast_h3::FastH3TransformerConfig;
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const RECIPE: &str = "fast_h3.vsa.logical_tile_positive_prefix_i32.v1";
#[derive(Clone, Copy, Debug)]
pub struct FastH3VsaPoolLimits {
    pub maximum_batches: u64,
    pub maximum_tiles: u64,
    pub maximum_columns: u64,
    pub maximum_count_bytes: u64,
    pub maximum_metadata_bytes: usize,
}
impl FastH3VsaPoolLimits {
    fn validate(self) -> Result<(), VNextError> {
        if [
            self.maximum_batches,
            self.maximum_tiles,
            self.maximum_columns,
            self.maximum_count_bytes,
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
pub struct FastH3VsaPoolConfig {
    batches: u64,
    heads: u64,
    head_dim: u64,
    tiles: u64,
    counts_sha256: String,
    component: WeightComponentSpec,
}
impl FastH3VsaPoolConfig {
    pub fn from_layout(
        layout: &FastH3VsaLayout,
        batches: u64,
        transformer: &FastH3TransformerConfig,
        limits: FastH3VsaPoolLimits,
    ) -> Result<Self, VNextError> {
        Self::build(
            layout,
            batches,
            transformer.attention_heads() as u64,
            transformer.attention_head_dim() as u64,
            limits,
        )
    }
    fn build(
        layout: &FastH3VsaLayout,
        batches: u64,
        heads: u64,
        head_dim: u64,
        limits: FastH3VsaPoolLimits,
    ) -> Result<Self, VNextError> {
        let tiles = u64::try_from(layout.tile_count()).map_err(invalid)?;
        if tiles == 0
            || tiles > limits.maximum_tiles
            || tiles
                .checked_mul(4)
                .is_none_or(|n| n > limits.maximum_count_bytes)
            || layout.tile_count().checked_mul(FAST_H3_VSA_TILE_ROWS) != Some(layout.padded_rows())
        {
            return Err(invalid("logical tile count/row capacity differs"));
        }
        let mut hash = Sha256::new();
        hash.update(RECIPE.as_bytes());
        hash.update((FAST_H3_VSA_TILE_ROWS as u64).to_le_bytes());
        hash.update(tiles.to_le_bytes());
        for &count in layout.valid_rows_per_tile() {
            if count == 0 || usize::from(count) > FAST_H3_VSA_TILE_ROWS {
                return Err(invalid(
                    "logical tiles require positive bounded valid prefixes",
                ));
            }
            hash.update(i32::from(count).to_le_bytes());
        }
        let counts_sha256 = format!("{:x}", hash.finalize());
        let c = Self {
            batches,
            heads,
            head_dim,
            tiles,
            component: component(tiles, &counts_sha256)?,
            counts_sha256,
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
    pub fn head_dim(&self) -> u64 {
        self.head_dim
    }
    pub fn tiles(&self) -> u64 {
        self.tiles
    }
    pub fn counts_sha256(&self) -> &str {
        &self.counts_sha256
    }
    pub fn input_shape(&self) -> Result<Vec<u64>, VNextError> {
        Ok(vec![
            self.batches,
            self.tiles
                .checked_mul(FAST_H3_VSA_TILE_ROWS as u64)
                .ok_or_else(|| invalid("row count overflows"))?,
            self.heads
                .checked_mul(self.head_dim)
                .ok_or_else(|| invalid("head width overflows"))?,
        ])
    }
    pub fn output_shape(&self) -> Vec<u64> {
        vec![self.batches, self.heads, self.tiles, self.head_dim]
    }
    pub fn compile_options(
        &self,
        limits: FastH3VsaPoolLimits,
    ) -> Result<ProgramPlanCompileOptions, VNextError> {
        self.validate(limits)?;
        ProgramPlanCompileOptions::new(self.specs()?)
    }
    fn validate(&self, limits: FastH3VsaPoolLimits) -> Result<(), VNextError> {
        limits.validate()?;
        if self.batches == 0
            || self.batches > limits.maximum_batches
            || self.tiles == 0
            || self.tiles > limits.maximum_tiles
            || self.heads == 0
            || self.head_dim == 0
            || self
                .heads
                .checked_mul(self.head_dim)
                .is_none_or(|n| n > limits.maximum_columns)
            || self.counts_sha256.len() != 64
            || !self
                .counts_sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            || self.component != component(self.tiles, &self.counts_sha256)?
            || self.component.physical_bytes()? > limits.maximum_count_bytes
            || serde_json::to_vec(self).map_err(invalid)?.len() > limits.maximum_metadata_bytes
        {
            return Err(invalid("invalid pool geometry, count identity or capacity"));
        }
        MaskedTileMeanBf16F32Geometry::new(
            &resolved(self.input_shape()?, ElementType::Bf16)?,
            &resolved(vec![self.tiles], ElementType::I32)?,
            &resolved(self.output_shape(), ElementType::F32)?,
            &MaskedTileMeanBf16F32Geometry::attributes(self.heads)?,
        )?;
        Ok(())
    }
    fn specs(&self) -> Result<BTreeMap<ProgramValueId, ProgramTensorSpec>, VNextError> {
        Ok(BTreeMap::from([
            (
                id(FastH3VsaPoolFamily::INPUT_ID)?,
                tensor(self.input_shape()?, ElementType::Bf16),
            ),
            (
                id(FastH3VsaPoolFamily::OUTPUT_ID)?,
                tensor(self.output_shape(), ElementType::F32),
            ),
        ]))
    }
    fn schema(&self) -> Result<WeightSchema, VNextError> {
        Ok(WeightSchema {
            format_id: id("weight-format.safetensors.dense")?,
            layout_id: id("weight-layout.fast_h3.vsa.positive_counts.v1")?,
            version: ContractVersion::new(1, 0),
            components: vec![self.component.clone()],
            tensors: vec![WeightTensorSpec {
                id: id("weight.fast_h3.vsa.pool.counts")?,
                dimensions: vec![self.tiles],
                logical_element_type: ElementType::I32,
                physical_layout: PhysicalWeightLayout::Dense {
                    component_id: self.component.id.clone(),
                },
                required: true,
            }],
        })
    }
}
pub struct FastH3VsaPoolSource {
    component: WeightComponentSpec,
    bytes: ManagedHostBytes,
    source_file: String,
}
impl FastH3VsaPoolSource {
    pub fn new(
        c: &FastH3VsaPoolConfig,
        layout: &FastH3VsaLayout,
        host: &HostMemoryAccount,
        limits: FastH3VsaPoolLimits,
    ) -> Result<Self, VNextError> {
        c.validate(limits)?;
        if Self::checked_config(c, layout, limits)? != *c {
            return Err(invalid(
                "pool counts differ from checked logical tile layout",
            ));
        }
        let mut bytes = host
            .allocate_zeroed(
                HostMemoryClass::WeightBacking,
                usize::try_from(c.component.physical_bytes()?).map_err(invalid)?,
            )
            .map_err(invalid)?;
        for (slot, &count) in bytes
            .as_mut()
            .chunks_exact_mut(4)
            .zip(layout.valid_rows_per_tile())
        {
            slot.copy_from_slice(&i32::from(count).to_le_bytes());
        }
        Ok(Self {
            component: c.component.clone(),
            bytes: bytes.freeze(),
            source_file: format!("derived/fast_h3/vsa/positive-counts/{}", c.counts_sha256),
        })
    }
    fn checked_config(
        c: &FastH3VsaPoolConfig,
        layout: &FastH3VsaLayout,
        limits: FastH3VsaPoolLimits,
    ) -> Result<FastH3VsaPoolConfig, VNextError> {
        FastH3VsaPoolConfig::build(layout, c.batches, c.heads, c.head_dim, limits)
    }
}
impl WeightComponentSource for FastH3VsaPoolSource {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        if c != &self.component {
            return Err(invalid("undeclared pool count component"));
        }
        WeightComponentPayload::from_managed_host(
            c,
            c.external_names.clone(),
            vec![self.source_file.clone()],
            c.dimensions.clone(),
            ElementType::I32,
            self.bytes.clone(),
        )
    }
}
pub struct FastH3VsaPoolFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: FastH3VsaPoolLimits,
}
impl FastH3VsaPoolFamily {
    pub const FAMILY_ID: &'static str = "family.fast_h3.vsa.pool";
    pub const PROFILE_ID: &'static str = "fast_h3.vsa.pool.f32_ascending_masked_rn.v1";
    pub const INPUT_ID: &'static str = "value.fast_h3.vsa.pool.tiled";
    pub const OUTPUT_ID: &'static str = "value.fast_h3.vsa.pool.pooled";
    pub fn new(limits: FastH3VsaPoolLimits) -> Result<Self, VNextError> {
        limits.validate()?;
        Ok(Self {
            id: id(Self::FAMILY_ID)?,
            metadata: id("metadata.fast_h3.vsa.pool")?,
            limits,
        })
    }
}
impl ModelFamilyProvider for FastH3VsaPoolFamily {
    type Config = FastH3VsaPoolConfig;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([self.metadata.clone()])
    }
    fn validate_config_identity(
        &self,
        r: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<(), VNextError> {
        if serde_json::to_value(c).map_err(invalid)? != *r {
            return Err(invalid("noncanonical pool config"));
        }
        c.validate(self.limits)
    }
    fn validated_external_metadata_id(
        &self,
        r: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(r, c)?;
        Ok(self.metadata.clone())
    }
    fn parse_config(&self, r: &serde_json::Value) -> Result<Self::Config, VNextError> {
        let c = serde_json::from_value(r.clone()).map_err(invalid)?;
        self.validate_config_identity(r, &c)?;
        Ok(c)
    }
    fn weights(&self, c: &Self::Config) -> Result<ModelWeights, VNextError> {
        c.validate(self.limits)?;
        Ok(ModelWeights::Stored(c.schema()?))
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
                boundaries: BTreeMap::from([(id(Self::OUTPUT_ID)?, ElementType::F32)]),
                states: vec![],
                kv_storage: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(MASKED_TILE_MEAN_BF16_F32_OPERATION_ID)?,
                    version: ContractVersion::new(1, 0),
                    multiplication_type: None,
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
            return Err(invalid("unknown pool profile"));
        }
        ModelProgram::new(
            self.id.clone(),
            vec![id(Self::INPUT_ID)?],
            vec![ProgramBlock {
                id: "block.fast_h3.vsa.pool".into(),
                nodes: vec![ProgramNode {
                    id: id("node.fast_h3.vsa.pool.mean")?,
                    operation_id: id(MASKED_TILE_MEAN_BF16_F32_OPERATION_ID)?,
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: vec![id(Self::INPUT_ID)?, id("value.fast_h3.vsa.pool.counts")?],
                    outputs: vec![id(Self::OUTPUT_ID)?],
                    attributes: MaskedTileMeanBf16F32Geometry::attributes(c.heads)?,
                }],
            }],
            vec![],
            vec![WeightReference {
                weight_id: id("weight.fast_h3.vsa.pool.counts")?,
                value_id: id("value.fast_h3.vsa.pool.counts")?,
                tensor: tensor(vec![c.tiles], ElementType::I32),
            }],
            vec![id(Self::OUTPUT_ID)?],
        )?
        .with_input_tensor_specs(BTreeMap::from([(
            id(Self::INPUT_ID)?,
            tensor(c.input_shape()?, ElementType::Bf16),
        )]))
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn component(tiles: u64, digest: &str) -> Result<WeightComponentSpec, VNextError> {
    Ok(WeightComponentSpec {
        id: id(&format!("derived.fast_h3.vsa.positive_counts.{digest}"))?,
        role: WeightComponentRole::Values,
        external_names: vec![format!("derived.vsa.positive_counts.{digest}")],
        dimensions: vec![tiles],
        encoding: WeightEncoding::Dense {
            element_type: ElementType::I32,
        },
        required: true,
    })
}
fn tensor(dimensions: Vec<u64>, element_type: ElementType) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
fn resolved(shape: Vec<u64>, dtype: ElementType) -> Result<ResolvedTensorSpec, VNextError> {
    ResolvedTensorSpec::new(shape, dtype, ResolvedTensorLayout::Contiguous)
}
fn id<T: TryFrom<String, Error = VNextError>>(s: &str) -> Result<T, VNextError> {
    T::try_from(s.to_owned())
}
fn invalid(e: impl std::fmt::Display) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("FastH3 VSA pool: {e}"),
    }
}
