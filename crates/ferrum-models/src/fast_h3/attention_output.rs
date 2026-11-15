//! Active VSA compression merge and the original NVFP4 output projection.
//! All inputs are already in packed document order. Pooling/attention and tile
//! routing remain explicit upstream producers; no clamp or gate activation is
//! introduced. This component does not certify a structurally inactive gate.
use super::{
    nvfp4_scalars::FastH3UnitAlphaDerivation, FastH3TransformerPackage, FastH3WeightStorage,
};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

mod program;
mod source;
pub use source::FastH3AttentionOutputSource;

#[derive(Clone, Copy, Debug)]
pub struct FastH3AttentionOutputLimits {
    pub maximum_rows: u64,
    pub maximum_metadata_bytes: usize,
    pub maximum_stored_bytes: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FastH3AttentionMerge {
    /// BF16 pooled output is expanded to rows before this component. Raw gate
    /// is BF16. Merge uses one F32 FMA, then BF16 RNE.
    ActiveCompressionFma,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastH3AttentionOutputConfig {
    rows: u64,
    block: usize,
    hidden: u64,
    heads: u64,
    head_dim: u64,
    merge: FastH3AttentionMerge,
    packed: WeightComponentSpec,
    scales: WeightComponentSpec,
    alpha: FastH3UnitAlphaDerivation,
}
impl FastH3AttentionOutputConfig {
    pub fn rows(&self) -> u64 {
        self.rows
    }
    pub fn block(&self) -> usize {
        self.block
    }
    pub fn hidden(&self) -> u64 {
        self.hidden
    }
    pub fn merge(&self) -> FastH3AttentionMerge {
        self.merge
    }
    pub fn scalar_derivation(&self) -> &FastH3UnitAlphaDerivation {
        &self.alpha
    }
    pub fn attention_width(&self) -> Result<u64, VNextError> {
        self.heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| invalid("attention width overflows"))
    }
    pub fn compile_options(&self) -> Result<ProgramPlanCompileOptions, VNextError> {
        ProgramPlanCompileOptions::new(program::build(self)?.1)
    }
    fn validate(&self, limits: FastH3AttentionOutputLimits) -> Result<(), VNextError> {
        limits.validate()?;
        if self.rows == 0
            || self.rows > limits.maximum_rows
            || self.rows > i64::MAX as u64
            || self.hidden == 0
            || self.heads == 0
            || self.head_dim == 0
        {
            return Err(invalid("invalid attention output geometry or row capacity"));
        }
        let matrix = Nvfp4MatrixLayout::new(self.hidden, self.attention_width()?)?;
        self.alpha.validate()?;
        if self.packed.role != WeightComponentRole::PackedValues
            || self.packed.encoding
                != WeightEncoding::Quantized(QuantizationSpec::nvfp4_e2m1_e4m3_128x4())
            || self.packed.dimensions != matrix.packed_shape()
            || self.scales.role != WeightComponentRole::Scales
            || self.scales.encoding
                != (WeightEncoding::Dense {
                    element_type: ElementType::U8,
                })
            || self.scales.physical_bytes()? != matrix.scale_bytes()
        {
            return Err(invalid(
                "original output projection encoding or shape differs",
            ));
        }
        for (c, suffix) in [
            (&self.packed, "_nvfp4_weight"),
            (&self.scales, "_nvfp4_weight_scale"),
            (self.alpha.source_alpha(), "_nvfp4_alpha"),
        ] {
            let expected = format!("transformer_blocks.{}.attn.to_out::{suffix}", self.block);
            if !c.required
                || c.external_names.len() != 1
                || c.external_names[0]
                    .strip_prefix("minimax_h3.")
                    .unwrap_or(&c.external_names[0])
                    != expected
            {
                return Err(invalid(
                    "source is not the declared block's output projection",
                ));
            }
        }
        if serde_json::to_vec(self).map_err(invalid)?.len() > limits.maximum_metadata_bytes {
            return Err(invalid("attention output metadata capacity exceeded"));
        }
        let bytes = self.schema()?.components.iter().try_fold(0u64, |sum, c| {
            sum.checked_add(c.physical_bytes()?)
                .ok_or_else(|| invalid("source byte count overflows"))
        })?;
        if bytes > limits.maximum_stored_bytes {
            return Err(invalid("attention output source capacity exceeded"));
        }
        for spec in program::build(self)?.1.values() {
            spec.byte_len()?;
        }
        Ok(())
    }
    fn schema(&self) -> Result<WeightSchema, VNextError> {
        let matrix = Nvfp4MatrixLayout::new(self.hidden, self.attention_width()?)?;
        let alpha = self.alpha.output_component()?;
        let unit = source::unit_component()?;
        let parts = [
            ("packed", &self.packed, matrix.packed_shape().to_vec(), true),
            ("scales", &self.scales, matrix.scale_shape().to_vec(), true),
            ("alpha", &alpha, vec![1], false),
            ("unit", &unit, vec![1], false),
        ];
        let schema = WeightSchema {
            format_id: id("weight-format.safetensors.dense")?,
            layout_id: id("weight-layout.fast_h3.attention_output.original_and_unit_alpha.v1")?,
            version: ContractVersion::new(1, 0),
            components: parts.iter().map(|(_, c, _, _)| (*c).clone()).collect(),
            tensors: parts
                .into_iter()
                .map(|(name, c, dimensions, physical)| {
                    Ok(WeightTensorSpec {
                        id: weight(name)?,
                        dimensions,
                        logical_element_type: c.physical_element_type(),
                        required: true,
                        physical_layout: if physical {
                            PhysicalWeightLayout::PhysicalComponent(
                                PhysicalWeightComponentView::new(c.id.clone()),
                            )
                        } else {
                            PhysicalWeightLayout::Dense {
                                component_id: c.id.clone(),
                            }
                        },
                    })
                })
                .collect::<Result<_, VNextError>>()?,
        };
        schema.validate(&id(FastH3AttentionOutputFamily::FAMILY_ID)?)?;
        Ok(schema)
    }
}
impl FastH3AttentionOutputLimits {
    fn validate(self) -> Result<(), VNextError> {
        if self.maximum_rows == 0
            || self.maximum_metadata_bytes == 0
            || self.maximum_stored_bytes == 0
        {
            Err(invalid("attention output limits must be positive"))
        } else {
            Ok(())
        }
    }
}

pub struct FastH3AttentionOutputFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: FastH3AttentionOutputLimits,
}
impl FastH3AttentionOutputFamily {
    pub const FAMILY_ID: &'static str = "family.fast_h3.attention_output.active_compression";
    pub const PROFILE_ID: &'static str = "fast_h3.attention_output.fma_bf16.unit_nvfp4.v1";
    pub const FINE_ID: &'static str = "value.fast_h3.attention_output.fine";
    pub const POOLED_ID: &'static str = "value.fast_h3.attention_output.expanded_pooled";
    pub const GATE_ID: &'static str = "value.fast_h3.attention_output.raw_gate";
    pub const OUTPUT_ID: &'static str = "value.fast_h3.attention_output.projected";
    pub fn new(limits: FastH3AttentionOutputLimits) -> Result<Self, VNextError> {
        limits.validate()?;
        Ok(Self {
            id: id(Self::FAMILY_ID)?,
            metadata: id("metadata.fast_h3.attention_output.active_compression")?,
            limits,
        })
    }
}
impl ModelFamilyProvider for FastH3AttentionOutputFamily {
    type Config = FastH3AttentionOutputConfig;
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
            return Err(invalid("noncanonical attention output configuration"));
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
        let (program, geometry) = program::build(c)?;
        let boundaries = program
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .flat_map(|n| &n.outputs)
            .map(|v| (v.clone(), geometry[v].element_type))
            .collect();
        let profile = NumericalExecutionProfile {
            id: NumericalProfileId::new(Self::PROFILE_ID).map_err(invalid)?,
            family_id: self.id.clone(),
            version: ContractVersion::new(1, 0),
            primary_activation: id(Self::OUTPUT_ID)?,
            boundaries,
            states: vec![],
            kv_storage: vec![],
            operations: [
                ADDCMUL_FMA_BF16_OPERATION_ID,
                NVFP4_QUANTIZE_BF16_OPERATION_ID,
                NVFP4_MATMUL_BF16_OPERATION_ID,
            ]
            .into_iter()
            .map(|op| {
                Ok(NumericalOperationContract {
                    operation_id: id(op)?,
                    version: ContractVersion::new(1, 0),
                    multiplication_type: matches!(
                        op,
                        ADDCMUL_FMA_BF16_OPERATION_ID | NVFP4_QUANTIZE_BF16_OPERATION_ID
                    )
                    .then_some(ElementType::F32),
                    accumulation_type: (op != NVFP4_QUANTIZE_BF16_OPERATION_ID)
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
            vec![profile.clone()],
            vec![profile.id],
        )
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        if self.numerical_profiles(c)?.resolve(&p.id)? != p {
            return Err(invalid("unknown attention output profile"));
        }
        Ok(program::build(c)?.0)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn id<T: TryFrom<String>>(s: &str) -> Result<T, VNextError>
where
    T::Error: std::fmt::Display,
{
    T::try_from(s.to_owned()).map_err(invalid)
}
fn value(s: &str) -> Result<ProgramValueId, VNextError> {
    id(&format!("value.fast_h3.attention_output.{s}"))
}
fn weight(s: &str) -> Result<WeightId, VNextError> {
    id(&format!("weight.fast_h3.attention_output.{s}"))
}
fn invalid(e: impl std::fmt::Display) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("FastH3 attention output: {e}"),
    }
}
