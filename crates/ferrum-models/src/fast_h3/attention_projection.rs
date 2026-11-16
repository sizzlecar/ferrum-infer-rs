//! Raw Q/K/V and compression-gate projections over already modulated BF16 rows.
//! Unit-scale activation quantization is shared once per explicit group. The
//! VSA path projects Q/K/V on tiled rows and gate on original packed rows;
//! AllFour remains the explicit diagnostic group. VSA gate selection, QK norm, RoPE,
//! attention and the output projection remain separate consumers.
use super::{FastH3TransformerPackage, FastH3WeightStorage};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

mod program;
mod source;
pub use source::{FastH3AttentionProjectionSource, FastH3UnitAlphaDerivation};

#[derive(Clone, Copy, Debug)]
pub struct FastH3AttentionProjectionLimits {
    /// Rows of this explicit tensor invocation, not a hidden chunk size.
    pub maximum_rows: u64,
    pub maximum_metadata_bytes: usize,
    /// Selected packed/scales plus invocation constants, before alignment.
    /// Full archive import bounds remain the transformer's package contract.
    pub maximum_stored_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FastH3AttentionProjection {
    Query,
    Key,
    Value,
    CompressionGate,
}
impl FastH3AttentionProjection {
    pub const ALL: [Self; 4] = [Self::Query, Self::Key, Self::Value, Self::CompressionGate];
    pub const fn module(self) -> &'static str {
        match self {
            Self::Query => "to_q",
            Self::Key => "to_k",
            Self::Value => "to_v",
            Self::CompressionGate => "to_gate_compress",
        }
    }
    pub const fn output_id(self) -> &'static str {
        match self {
            Self::Query => "value.fast_h3.attention_projection.query",
            Self::Key => "value.fast_h3.attention_projection.key",
            Self::Value => "value.fast_h3.attention_projection.value",
            Self::CompressionGate => "value.fast_h3.attention_projection.compression_gate",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FastH3AttentionProjectionGroup {
    #[default]
    AllFour,
    Qkv,
    Gate,
}
impl FastH3AttentionProjectionGroup {
    pub fn projections(self) -> &'static [FastH3AttentionProjection] {
        use FastH3AttentionProjection::*;
        match self {
            Self::AllFour => &FastH3AttentionProjection::ALL,
            Self::Qkv => &[Query, Key, Value],
            Self::Gate => &[CompressionGate],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastH3AttentionProjectionConfig {
    rows: u64,
    block: usize,
    hidden: u64,
    heads: u64,
    head_dim: u64,
    #[serde(default)]
    group: FastH3AttentionProjectionGroup,
    projections: Vec<Projection>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Projection {
    kind: FastH3AttentionProjection,
    packed: WeightComponentSpec,
    scales: WeightComponentSpec,
    alpha: FastH3UnitAlphaDerivation,
}

impl FastH3AttentionProjectionConfig {
    pub fn group(&self) -> FastH3AttentionProjectionGroup {
        self.group
    }
    /// Select from an already checked source inventory without allocating
    /// another copy of its unit scalar or alpha backing.
    pub(crate) fn select_group(
        &self,
        group: FastH3AttentionProjectionGroup,
        rows: u64,
        limits: FastH3AttentionProjectionLimits,
    ) -> Result<Self, VNextError> {
        let mut c = self.clone();
        c.group = group;
        c.rows = rows;
        c.projections = group
            .projections()
            .iter()
            .map(|kind| {
                self.projections
                    .iter()
                    .find(|p| p.kind == *kind)
                    .cloned()
                    .ok_or_else(|| invalid("selected projection is absent from source group"))
            })
            .collect::<Result<_, _>>()?;
        c.validate(limits)?;
        Ok(c)
    }
    pub fn rows(&self) -> u64 {
        self.rows
    }
    pub fn block(&self) -> usize {
        self.block
    }
    pub fn hidden(&self) -> u64 {
        self.hidden
    }
    pub fn heads(&self) -> u64 {
        self.heads
    }
    pub fn head_dim(&self) -> u64 {
        self.head_dim
    }
    pub fn output_width(&self) -> Result<u64, VNextError> {
        self.heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| invalid("attention width overflows"))
    }
    pub fn scalar_derivations(&self) -> impl Iterator<Item = &FastH3UnitAlphaDerivation> {
        self.projections.iter().map(|p| &p.alpha)
    }
    pub fn compile_options(&self) -> Result<ProgramPlanCompileOptions, VNextError> {
        ProgramPlanCompileOptions::new(program::build(self)?.1)
    }
    fn validate(&self, limits: FastH3AttentionProjectionLimits) -> Result<(), VNextError> {
        if limits.maximum_rows == 0
            || limits.maximum_metadata_bytes == 0
            || limits.maximum_stored_bytes == 0
            || self.rows == 0
            || self.rows > limits.maximum_rows
            || self.rows > i64::MAX as u64
            || self.hidden == 0
            || self.heads == 0
            || self.head_dim == 0
        {
            return Err(invalid("invalid projection geometry or capacity"));
        }
        let geometry = Nvfp4MatrixLayout::new(self.output_width()?, self.hidden)?;
        if self.projections.len() != self.group.projections().len() {
            return Err(invalid("projection inventory differs from selected group"));
        }
        for (&kind, p) in self.group.projections().iter().zip(&self.projections) {
            p.alpha.validate()?;
            if p.kind != kind
                || p.packed.role != WeightComponentRole::PackedValues
                || p.packed.encoding
                    != WeightEncoding::Quantized(QuantizationSpec::nvfp4_e2m1_e4m3_128x4())
                || p.packed.dimensions != geometry.packed_shape()
                || p.scales.role != WeightComponentRole::Scales
                || p.scales.encoding
                    != (WeightEncoding::Dense {
                        element_type: ElementType::U8,
                    })
                || p.scales.physical_bytes()? != geometry.scale_bytes()
            {
                return Err(invalid("projection kind, physical layout or shape differs"));
            }
            for (c, suffix) in [
                (&p.packed, "_nvfp4_weight"),
                (&p.scales, "_nvfp4_weight_scale"),
                (p.alpha.source_alpha(), "_nvfp4_alpha"),
            ] {
                let expected = format!(
                    "transformer_blocks.{}.attn.{}::{suffix}",
                    self.block,
                    kind.module()
                );
                if !c.required
                    || c.external_names.len() != 1
                    || c.external_names[0]
                        .strip_prefix("minimax_h3.")
                        .unwrap_or(&c.external_names[0])
                        != expected
                {
                    return Err(invalid(
                        "source does not belong to the declared block projection",
                    ));
                }
            }
        }
        if serde_json::to_vec(self).map_err(invalid)?.len() > limits.maximum_metadata_bytes {
            return Err(invalid("projection metadata exceeds capacity"));
        }
        let schema = self.schema()?;
        let bytes = schema.components.iter().try_fold(0u64, |sum, c| {
            sum.checked_add(c.physical_bytes()?)
                .ok_or_else(|| invalid("source byte count overflows"))
        })?;
        if bytes > limits.maximum_stored_bytes {
            return Err(invalid("stored source exceeds capacity"));
        }
        for spec in program::build(self)?.1.values() {
            spec.byte_len()?;
        }
        Ok(())
    }
    fn schema(&self) -> Result<WeightSchema, VNextError> {
        let geometry = Nvfp4MatrixLayout::new(self.output_width()?, self.hidden)?;
        let unit = source::unit_component()?;
        let mut components = vec![unit.clone()];
        let mut tensors = vec![weight_tensor(weight("unit_sf")?, &unit, vec![1], false)];
        for p in &self.projections {
            let alpha = p.alpha.output_component()?;
            for (suffix, c, shape, physical) in [
                ("packed", &p.packed, geometry.packed_shape().to_vec(), true),
                ("scales", &p.scales, geometry.scale_shape().to_vec(), true),
                ("alpha", &alpha, vec![1], false),
            ] {
                components.push(c.clone());
                tensors.push(weight_tensor(
                    weight(&format!("{}.{}", p.kind.module(), suffix))?,
                    c,
                    shape,
                    physical,
                ));
            }
        }
        let s = WeightSchema {
            format_id: WeightFormatId::new("weight-format.safetensors.dense")?,
            layout_id: WeightLayoutId::new(
                "weight-layout.fast_h3.attention_projection.unit.original_and_derived_alpha.v1",
            )?,
            version: ContractVersion::new(1, 0),
            components,
            tensors,
        };
        s.validate(&ModelFamilyId::new(
            FastH3AttentionProjectionFamily::FAMILY_ID,
        )?)?;
        Ok(s)
    }
}

fn weight_tensor(
    id: WeightId,
    c: &WeightComponentSpec,
    dimensions: Vec<u64>,
    physical: bool,
) -> WeightTensorSpec {
    WeightTensorSpec {
        id,
        dimensions,
        logical_element_type: c.physical_element_type(),
        required: true,
        physical_layout: if physical {
            PhysicalWeightLayout::PhysicalComponent(PhysicalWeightComponentView::new(c.id.clone()))
        } else {
            PhysicalWeightLayout::Dense {
                component_id: c.id.clone(),
            }
        },
    }
}

pub struct FastH3AttentionProjectionFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: FastH3AttentionProjectionLimits,
}
impl FastH3AttentionProjectionFamily {
    pub const FAMILY_ID: &'static str = "family.fast_h3.attention_projection.unit";
    pub const PROFILE_ID: &'static str =
        "fast_h3.attention_projection.unit.shared_quant.f32_alpha.v1";
    pub const INPUT_ID: &'static str = "value.fast_h3.attention_projection.input";
    pub fn new(limits: FastH3AttentionProjectionLimits) -> Result<Self, VNextError> {
        if limits.maximum_rows == 0
            || limits.maximum_metadata_bytes == 0
            || limits.maximum_stored_bytes == 0
        {
            return Err(invalid("projection limits must be positive"));
        }
        Ok(Self {
            id: ModelFamilyId::new(Self::FAMILY_ID)?,
            metadata: ExternalModelMetadataId::new("metadata.fast_h3.attention_projection.unit")?,
            limits,
        })
    }
}
impl ModelFamilyProvider for FastH3AttentionProjectionFamily {
    type Config = FastH3AttentionProjectionConfig;
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
            return Err(invalid("noncanonical projection configuration"));
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
        let (program, specs) = program::build(c)?;
        let profile = NumericalExecutionProfile {
            id: NumericalProfileId::new(Self::PROFILE_ID).map_err(invalid)?,
            family_id: self.id.clone(),
            version: ContractVersion::new(1, 0),
            primary_activation: ProgramValueId::new(c.group.projections()[0].output_id())?,
            boundaries: program
                .blocks()
                .iter()
                .flat_map(|block| &block.nodes)
                .flat_map(|node| &node.outputs)
                .map(|value| (value.clone(), specs[value].element_type))
                .collect(),
            states: vec![],
            kv_storage: vec![],
            operations: [
                NVFP4_QUANTIZE_BF16_OPERATION_ID,
                NVFP4_MATMUL_BF16_OPERATION_ID,
            ]
            .into_iter()
            .map(|op| {
                Ok(NumericalOperationContract {
                    operation_id: OperationId::new(op)?,
                    version: ContractVersion::new(1, 0),
                    multiplication_type: (op == NVFP4_QUANTIZE_BF16_OPERATION_ID)
                        .then_some(ElementType::F32),
                    accumulation_type: (op == NVFP4_MATMUL_BF16_OPERATION_ID)
                        .then_some(ElementType::F32),
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                })
            })
            .collect::<Result<_, VNextError>>()?,
        };
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            vec![profile],
            vec![NumericalProfileId::new(Self::PROFILE_ID).map_err(invalid)?],
        )
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        if self.numerical_profiles(c)?.resolve(&p.id)? != p {
            return Err(invalid("unknown projection profile"));
        }
        Ok(program::build(c)?.0)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn weight(suffix: &str) -> Result<WeightId, VNextError> {
    WeightId::new(format!("weight.fast_h3.attention_projection.{suffix}"))
}
fn value(suffix: &str) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!("value.fast_h3.attention_projection.{suffix}"))
}
fn invalid(reason: impl std::fmt::Display) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("FastH3 attention projection: {reason}"),
    }
}
