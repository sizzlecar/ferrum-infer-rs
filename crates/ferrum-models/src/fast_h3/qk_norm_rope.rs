//! Actual Q/K norm weights applied to projected heads. Position production is
//! separate: cosine/sine are explicit BF16 row tables, shared across heads.
//!
//! FastVideo `minimax_h3_fusions/qknorm_rope.py` keeps both normalized operands
//! in F32 through rotary. The explicit block128/warp4 profile follows the
//! inspected `_qknorm_partial_rope_kernel` PTX, including the first reduction
//! FMA and the partner/rotary FMAs. This family does not select the older
//! staged-BF16 P-H3 operation. CUDA approximate rsqrt and Reference libm sqrt
//! have distinct provider identities; sharing a model profile does not claim
//! bitwise equivalence between them. Position-table arithmetic is supplied by
//! its producer, independently of this operation's arithmetic profile.
use super::{FastH3TransformerPackage, FastH3WeightStorage};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug)]
pub struct FastH3QkNormRopeLimits {
    pub maximum_rows: u64,
    pub maximum_metadata_bytes: usize,
    pub maximum_stored_bytes: u64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastH3QkNormRopeConfig {
    block: usize,
    rows: u64,
    heads: u64,
    head_dim: u64,
    rotary_dim: u64,
    epsilon: CanonicalRational,
    gamma: [WeightComponentSpec; 2],
}
impl FastH3QkNormRopeConfig {
    pub fn from_package(
        package: &FastH3TransformerPackage,
        block: usize,
        rows: u64,
        limits: FastH3QkNormRopeLimits,
    ) -> Result<Self, VNextError> {
        let config = package.config();
        let c = Self {
            block,
            rows,
            heads: config.attention_heads().try_into().map_err(invalid)?,
            head_dim: config.attention_head_dim().try_into().map_err(invalid)?,
            rotary_dim: config.rotary_dim().try_into().map_err(invalid)?,
            epsilon: CanonicalRational::from_decimal_str(&config.qk_norm_eps().to_string())?,
            gamma: [gamma(package, block, "q")?, gamma(package, block, "k")?],
        };
        c.validate(limits)?;
        Ok(c)
    }
    pub fn block(&self) -> usize {
        self.block
    }
    pub fn rows(&self) -> u64 {
        self.rows
    }
    pub fn heads(&self) -> u64 {
        self.heads
    }
    pub fn head_dim(&self) -> u64 {
        self.head_dim
    }
    pub fn rotary_dim(&self) -> u64 {
        self.rotary_dim
    }
    pub(in crate::fast_h3) fn numerical_geometry(&self) -> (u64, u64, u64, CanonicalRational) {
        (self.heads, self.head_dim, self.rotary_dim, self.epsilon)
    }
    pub fn compile_options(&self) -> Result<ProgramPlanCompileOptions, VNextError> {
        ProgramPlanCompileOptions::new(self.program()?.1)
    }
    fn width(&self) -> Result<u64, VNextError> {
        self.heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| invalid("head extent overflow"))
    }
    fn attrs(&self) -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
        let mut a = HeadRmsPartialRopeSingleRoundBf16Profile::Block128Warp4Fma.attributes()?;
        for (key, value) in [
            ("head_dim", SemanticValue::Unsigned(self.head_dim)),
            ("rotary_dim", SemanticValue::Unsigned(self.rotary_dim)),
            ("epsilon", SemanticValue::Rational(self.epsilon)),
        ] {
            a.insert(id(key)?, value);
        }
        Ok(a)
    }
    fn validate(&self, limits: FastH3QkNormRopeLimits) -> Result<(), VNextError> {
        if self.rows == 0 || self.rows > limits.maximum_rows || self.heads == 0 {
            return Err(invalid("invalid rows or head count"));
        }
        let mut bytes = 0u64;
        for c in &self.gamma {
            if c.dimensions != [self.head_dim]
                || c.role != WeightComponentRole::Values
                || c.encoding
                    != (WeightEncoding::Dense {
                        element_type: ElementType::Bf16,
                    })
                || !c.required
                || c.external_names.len() != 1
            {
                return Err(invalid("requires original dense BF16 norm vectors"));
            }
            bytes = bytes
                .checked_add(c.physical_bytes()?)
                .ok_or_else(|| invalid("norm bytes overflow"))?;
        }
        if bytes > limits.maximum_stored_bytes
            || serde_json::to_vec(self).map_err(invalid)?.len() > limits.maximum_metadata_bytes
        {
            return Err(invalid("source or metadata capacity exceeded"));
        }
        let x = resolved(vec![self.rows, self.width()?])?;
        let g = resolved(vec![self.head_dim])?;
        let table = resolved(vec![self.rows, self.rotary_dim])?;
        HeadRmsPartialRopeSingleRoundBf16Geometry::new(&x, &g, &table, &table, &x, &self.attrs()?)?;
        self.schema()?
            .validate(&id(FastH3QkNormRopeFamily::FAMILY_ID)?)
    }
    fn schema(&self) -> Result<WeightSchema, VNextError> {
        Ok(WeightSchema {
            format_id: id("weight-format.safetensors.dense")?,
            layout_id: id("weight-layout.fast_h3.qk_norm_rope.v1")?,
            version: ContractVersion::new(1, 0),
            components: self.gamma.to_vec(),
            tensors: self
                .gamma
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    Ok(WeightTensorSpec {
                        id: id(&format!("weight.fast_h3.qk_norm_rope.gamma{i}"))?,
                        dimensions: vec![self.head_dim],
                        logical_element_type: ElementType::Bf16,
                        physical_layout: PhysicalWeightLayout::Dense {
                            component_id: c.id.clone(),
                        },
                        required: true,
                    })
                })
                .collect::<Result<_, VNextError>>()?,
        })
    }
    fn program(
        &self,
    ) -> Result<(ModelProgram, BTreeMap<ProgramValueId, ProgramTensorSpec>), VNextError> {
        let external = [
            FastH3QkNormRopeFamily::QUERY_ID,
            FastH3QkNormRopeFamily::KEY_ID,
            FastH3QkNormRopeFamily::COSINE_ID,
            FastH3QkNormRopeFamily::SINE_ID,
        ];
        let outputs = [
            FastH3QkNormRopeFamily::QUERY_OUTPUT_ID,
            FastH3QkNormRopeFamily::KEY_OUTPUT_ID,
        ];
        let mut specs = BTreeMap::new();
        for (i, name) in external.iter().enumerate() {
            specs.insert(
                id(name)?,
                tensor(vec![
                    self.rows,
                    if i < 2 {
                        self.width()?
                    } else {
                        self.rotary_dim
                    },
                ]),
            );
        }
        let inputs = specs.clone();
        let mut nodes = Vec::with_capacity(2);
        let mut weights = Vec::with_capacity(2);
        for i in 0..2 {
            let g: ProgramValueId = id(&format!("value.fast_h3.qk_norm_rope.gamma{i}"))?;
            nodes.push(ProgramNode {
                id: id(&format!("node.fast_h3.qk_norm_rope.{i}"))?,
                operation_id: id(HEAD_RMS_PARTIAL_ROPE_SINGLE_ROUND_BF16_OPERATION_ID)?,
                required_version: ContractVersion::new(1, 0),
                work: ProgramNodeWorkSpec::Fixed,
                inputs: vec![
                    id(external[i])?,
                    g.clone(),
                    id(external[2])?,
                    id(external[3])?,
                ],
                outputs: vec![id(outputs[i])?],
                attributes: self.attrs()?,
            });
            weights.push(WeightReference {
                weight_id: id(&format!("weight.fast_h3.qk_norm_rope.gamma{i}"))?,
                value_id: g,
                tensor: tensor(vec![self.head_dim]),
            });
            specs.insert(id(outputs[i])?, tensor(vec![self.rows, self.width()?]));
        }
        let program = ModelProgram::new(
            id(FastH3QkNormRopeFamily::FAMILY_ID)?,
            external.iter().map(|v| id(v)).collect::<Result<_, _>>()?,
            vec![ProgramBlock {
                id: "block.fast_h3.qk_norm_rope".into(),
                nodes,
            }],
            vec![],
            weights,
            outputs.iter().map(|v| id(v)).collect::<Result<_, _>>()?,
        )?
        .with_input_tensor_specs(inputs)?;
        Ok((program, specs))
    }
}
fn gamma(
    package: &FastH3TransformerPackage,
    block: usize,
    axis: &str,
) -> Result<WeightComponentSpec, VNextError> {
    if block >= package.config().layers() {
        return Err(invalid("block outside checkpoint"));
    }
    let w = package
        .weight(&format!(
            "transformer_blocks.{block}.attn.norm_{axis}.weight"
        ))
        .ok_or_else(|| invalid("missing Q/K norm"))?;
    let FastH3WeightStorage::Dense(c) = &w.storage else {
        return Err(invalid("Q/K norm must remain dense"));
    };
    if w.compute_element_type != ElementType::Bf16 {
        return Err(invalid("Q/K norm compute precision differs"));
    }
    Ok(c.clone())
}
pub struct FastH3QkNormRopeSource<'a> {
    package: &'a FastH3TransformerPackage,
    gamma: [WeightComponentSpec; 2],
}
impl<'a> FastH3QkNormRopeSource<'a> {
    pub fn new(
        package: &'a FastH3TransformerPackage,
        c: &FastH3QkNormRopeConfig,
        limits: FastH3QkNormRopeLimits,
    ) -> Result<Self, VNextError> {
        if &FastH3QkNormRopeConfig::from_package(package, c.block, c.rows, limits)? != c {
            return Err(invalid("source differs from checkpoint"));
        }
        Ok(Self {
            package,
            gamma: c.gamma.clone(),
        })
    }
}
impl WeightComponentSource for FastH3QkNormRopeSource<'_> {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        if !self.gamma.contains(c) {
            return Err(invalid("undeclared Q/K norm component"));
        }
        self.package.dense_archive().component(c)
    }
}
pub struct FastH3QkNormRopeFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: FastH3QkNormRopeLimits,
}
impl FastH3QkNormRopeFamily {
    pub const FAMILY_ID: &'static str = "family.fast_h3.qk_norm_rope";
    pub const PROFILE_ID: &'static str =
        "fast_h3.qk_norm_rope.block128_warp4_fma.single_round_bf16.v1";
    pub const QUERY_ID: &'static str = "value.fast_h3.qk_norm_rope.query";
    pub const KEY_ID: &'static str = "value.fast_h3.qk_norm_rope.key";
    pub const COSINE_ID: &'static str = "value.fast_h3.qk_norm_rope.cosine";
    pub const SINE_ID: &'static str = "value.fast_h3.qk_norm_rope.sine";
    pub const QUERY_OUTPUT_ID: &'static str = "value.fast_h3.qk_norm_rope.query_output";
    pub const KEY_OUTPUT_ID: &'static str = "value.fast_h3.qk_norm_rope.key_output";
    pub fn new(limits: FastH3QkNormRopeLimits) -> Result<Self, VNextError> {
        if limits.maximum_rows == 0
            || limits.maximum_metadata_bytes == 0
            || limits.maximum_stored_bytes == 0
        {
            return Err(invalid("capacities must be positive"));
        }
        Ok(Self {
            id: id(Self::FAMILY_ID)?,
            metadata: id("metadata.fast_h3.qk_norm_rope")?,
            limits,
        })
    }
}
impl ModelFamilyProvider for FastH3QkNormRopeFamily {
    type Config = FastH3QkNormRopeConfig;
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
            return Err(invalid("noncanonical Q/K configuration"));
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
        Ok(ModelWeights::Stored(c.schema()?))
    }
    fn numerical_profiles(&self, c: &Self::Config) -> Result<FamilyNumericalProfiles, VNextError> {
        c.validate(self.limits)?;
        let p = NumericalExecutionProfile {
            id: NumericalProfileId::new(Self::PROFILE_ID).map_err(invalid)?,
            family_id: self.id.clone(),
            version: ContractVersion::new(1, 0),
            primary_activation: id(Self::QUERY_OUTPUT_ID)?,
            boundaries: BTreeMap::from([
                (id(Self::QUERY_OUTPUT_ID)?, ElementType::Bf16),
                (id(Self::KEY_OUTPUT_ID)?, ElementType::Bf16),
            ]),
            states: vec![],
            kv_storage: vec![],
            operations: vec![NumericalOperationContract {
                operation_id: id(HEAD_RMS_PARTIAL_ROPE_SINGLE_ROUND_BF16_OPERATION_ID)?,
                version: ContractVersion::new(1, 0),
                multiplication_type: Some(ElementType::F32),
                accumulation_type: Some(ElementType::F32),
                staged_arithmetic: None,
                composite_arithmetic: None,
            }],
        };
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            vec![p],
            vec![NumericalProfileId::new(Self::PROFILE_ID).map_err(invalid)?],
        )
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        if self.numerical_profiles(c)?.resolve(&p.id)? != p {
            return Err(invalid("unknown Q/K profile"));
        }
        Ok(c.program()?.0)
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
fn resolved(dimensions: Vec<u64>) -> Result<ResolvedTensorSpec, VNextError> {
    ResolvedTensorSpec::new(
        dimensions,
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
}
fn id<T: TryFrom<String, Error = VNextError>>(s: &str) -> Result<T, VNextError> {
    T::try_from(s.to_owned())
}
fn invalid(e: impl std::fmt::Display) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("FastH3 Q/K norm/RoPE: {e}"),
    }
}
