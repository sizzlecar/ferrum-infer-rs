//! One FastH3 bias-free static-QAD FFN as a managed tensor component.
//! This composes existing BF16/gather/GEMM operations; it is not a transformer
//! block, scheduler, or registered video pipeline.
//!
//! Pinned FastVideo sources: `models/dits/minimax_h3.py`'s
//! `MiniMaxH3FeedForward`, `minimax_h3_fusions/swiglu.py`'s fused kernel,
//! and `layers/quantization/nvfp4_config.py`'s stored-alpha division. The graph
//! explicitly selects the CUDA reciprocal/FTZ quantizer profile. Provider
//! admission and device numerical qualification remain separate requirements.
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

mod program;
mod source;
pub use source::{FastH3FfnSource, FastH3ScalarDerivation};

const SCALAR_PROFILE: &str = "fast_h3.nvfp4.stored_alpha_div_static_sf.rn_f32.v1";

#[derive(Clone, Copy, Debug)]
pub struct FastH3FfnLimits {
    /// Packed rows in this one graph invocation, not an implicit chunk size.
    pub maximum_rows: u64,
    pub maximum_plan_nodes: usize,
    pub maximum_metadata_bytes: usize,
    /// Selected physical components, excluding unselected transformer weights.
    pub maximum_stored_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastH3FfnConfig {
    rows: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    chunk_rows: Option<std::num::NonZeroU64>,
    block: usize,
    hidden: u64,
    intermediate: u64,
    projections: [Projection; 2],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Projection {
    rows: u64,
    columns: u64,
    packed: WeightComponentSpec,
    scales: WeightComponentSpec,
    global_sf: WeightComponentSpec,
    alpha: WeightComponentSpec,
    derivation: FastH3ScalarDerivation,
}

impl FastH3FfnConfig {
    /// Row partitioning preserves the original weights and every arithmetic
    /// boundary. It does not partition reductions within a row.
    pub fn with_chunk_rows(mut self, maximum_rows: Option<std::num::NonZeroU64>) -> Self {
        self.chunk_rows = maximum_rows;
        self
    }
    pub fn chunk_rows(&self) -> Option<std::num::NonZeroU64> {
        self.chunk_rows
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
    pub fn intermediate(&self) -> u64 {
        self.intermediate
    }
    pub fn scalar_derivations(&self) -> impl Iterator<Item = &FastH3ScalarDerivation> {
        self.projections.iter().map(|p| &p.derivation)
    }
    pub fn compile_options(&self) -> Result<ProgramPlanCompileOptions, VNextError> {
        let (_, specs) = program::build(self)?;
        ProgramPlanCompileOptions::new(specs)
    }
    fn validate(&self, limits: FastH3FfnLimits) -> Result<(), VNextError> {
        if limits.maximum_rows == 0
            || limits.maximum_plan_nodes == 0
            || limits.maximum_metadata_bytes == 0
            || limits.maximum_stored_bytes == 0
            || self.rows == 0
            || self.rows > limits.maximum_rows
            || self.rows > i64::MAX as u64
            || self.hidden == 0
            || self.intermediate == 0
        {
            return Err(invalid("invalid FFN geometry or capacity"));
        }
        let required_nodes = program::node_count(self)?;
        if required_nodes > limits.maximum_plan_nodes as u64 {
            return Err(VNextError::ModelProgramNodeCapacityExceeded {
                required_nodes,
                maximum_nodes: limits.maximum_plan_nodes as u64,
            });
        }
        let doubled = self
            .intermediate
            .checked_mul(2)
            .ok_or_else(|| invalid("FFN width overflows"))?;
        for (index, (projection, (n, k))) in self
            .projections
            .iter()
            .zip([(doubled, self.hidden), (self.hidden, self.intermediate)])
            .enumerate()
        {
            if (projection.rows, projection.columns) != (n, k) {
                return Err(invalid("FFN projections do not cover the declared branch"));
            }
            let geometry = Nvfp4MatrixLayout::new(n, k)?;
            if projection.packed.role != WeightComponentRole::PackedValues
                || projection.packed.encoding
                    != WeightEncoding::Quantized(QuantizationSpec::nvfp4_e2m1_e4m3_128x4())
                || projection.packed.dimensions != geometry.packed_shape()
                || projection.scales.role != WeightComponentRole::Scales
                || projection.scales.encoding
                    != (WeightEncoding::Dense {
                        element_type: ElementType::U8,
                    })
                || projection.scales.physical_bytes()? != geometry.scale_bytes()
                || projection.global_sf != projection.derivation.source_global_sf
                || projection.alpha != projection.derivation.output_component()?
            {
                return Err(invalid(
                    "FFN source shape, encoding or scalar derivation differs",
                ));
            }
            projection.derivation.validate()?;
            let projection_name = if index == 0 { "fc_in" } else { "fc_out" };
            for (component, suffix) in [
                (&projection.packed, "_nvfp4_weight"),
                (&projection.scales, "_nvfp4_weight_scale"),
                (&projection.global_sf, "_nvfp4_input_global_sf"),
                (&projection.derivation.source_alpha, "_nvfp4_alpha"),
            ] {
                let expected = format!(
                    "transformer_blocks.{}.ff.{projection_name}::{suffix}",
                    self.block
                );
                if component.external_names.len() != 1
                    || component.external_names[0]
                        .strip_prefix("minimax_h3.")
                        .unwrap_or(&component.external_names[0])
                        != expected
                {
                    return Err(invalid(
                        "FFN source does not belong to the declared block projection",
                    ));
                }
            }
        }
        let metadata = serde_json::to_vec(self).map_err(|e| invalid(e.to_string()))?;
        if metadata.len() > limits.maximum_metadata_bytes {
            return Err(invalid("FFN metadata exceeds capacity"));
        }
        let schema = self.schema()?;
        let bytes = schema.components.iter().try_fold(0u64, |sum, c| {
            sum.checked_add(c.physical_bytes()?)
                .ok_or_else(|| invalid("FFN stored byte count overflows"))
        })?;
        if bytes > limits.maximum_stored_bytes {
            return Err(invalid("FFN stored weights exceed capacity"));
        }
        for spec in program::build(self)?.1.values() {
            spec.byte_len()?;
        }
        Ok(())
    }
    fn schema(&self) -> Result<WeightSchema, VNextError> {
        let mut components = Vec::new();
        let mut tensors = Vec::new();
        for (index, p) in self.projections.iter().enumerate() {
            let geometry = Nvfp4MatrixLayout::new(p.rows, p.columns)?;
            for (suffix, c, shape, physical) in [
                ("packed", &p.packed, geometry.packed_shape().to_vec(), true),
                ("scales", &p.scales, geometry.scale_shape().to_vec(), true),
                ("global_sf", &p.global_sf, vec![1], true),
                ("alpha", &p.alpha, vec![1], false),
            ] {
                components.push(c.clone());
                tensors.push(WeightTensorSpec {
                    id: weight(index, suffix)?,
                    dimensions: shape,
                    logical_element_type: c.physical_element_type(),
                    physical_layout: if physical {
                        PhysicalWeightLayout::PhysicalComponent(PhysicalWeightComponentView::new(
                            c.id.clone(),
                        ))
                    } else {
                        PhysicalWeightLayout::Dense {
                            component_id: c.id.clone(),
                        }
                    },
                    required: true,
                });
            }
        }
        let schema = WeightSchema {
            format_id: WeightFormatId::new("weight-format.safetensors.dense")?,
            layout_id: WeightLayoutId::new(
                "weight-layout.fast_h3.ffn.static_qad.original_and_derived_alpha.v1",
            )?,
            version: ContractVersion::new(1, 0),
            components,
            tensors,
        };
        schema.validate(&ModelFamilyId::new(FastH3FfnFamily::FAMILY_ID)?)?;
        Ok(schema)
    }
}

pub struct FastH3FfnFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: FastH3FfnLimits,
}
impl FastH3FfnFamily {
    pub const FAMILY_ID: &'static str = "family.fast_h3.ffn.static_qad";
    pub const PROFILE_ID: &'static str = "fast_h3.ffn.static_qad.exp2_sigmoid.single_bf16.v1";
    pub const INPUT_ID: &'static str = "value.fast_h3.ffn.input";
    pub const OUTPUT_ID: &'static str = "value.fast_h3.ffn.output";
    pub fn new(limits: FastH3FfnLimits) -> Result<Self, VNextError> {
        if limits.maximum_rows == 0
            || limits.maximum_plan_nodes == 0
            || limits.maximum_metadata_bytes == 0
            || limits.maximum_stored_bytes == 0
        {
            return Err(invalid("FFN limits must be positive"));
        }
        Ok(Self {
            id: ModelFamilyId::new(Self::FAMILY_ID)?,
            metadata: ExternalModelMetadataId::new("metadata.fast_h3.ffn.static_qad")?,
            limits,
        })
    }
}
impl ModelFamilyProvider for FastH3FfnFamily {
    type Config = FastH3FfnConfig;
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
        if serde_json::to_value(c).map_err(|e| invalid(e.to_string()))? != *raw {
            return Err(invalid("FFN config must be canonical"));
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
        let c = serde_json::from_value(raw.clone()).map_err(|e| invalid(e.to_string()))?;
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
        let operations = program
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .map(|n| n.operation_id.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|operation_id| {
                let name = operation_id.as_str();
                NumericalOperationContract {
                    multiplication_type: matches!(
                        name,
                        NVFP4_QUANTIZE_BF16_OPERATION_ID | PACKED_SWIGLU_BF16_OPERATION_ID
                    )
                    .then_some(ElementType::F32),
                    accumulation_type: (name == NVFP4_MATMUL_BF16_OPERATION_ID)
                        .then_some(ElementType::F32),
                    operation_id,
                    version: ContractVersion::new(1, 0),
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                }
            })
            .collect();
        let boundaries = program
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .flat_map(|n| &n.outputs)
            .map(|v| (v.clone(), specs[v].element_type))
            .collect();
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: NumericalProfileId::new(Self::PROFILE_ID).map_err(invalid)?,
                family_id: self.id.clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: ProgramValueId::new(Self::OUTPUT_ID)?,
                boundaries,
                states: vec![],
                kv_storage: vec![],
                operations,
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
            return Err(invalid("unknown FFN numerical profile"));
        }
        Ok(program::build(c)?.0)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn weight(index: usize, suffix: &str) -> Result<WeightId, VNextError> {
    WeightId::new(format!("weight.fast_h3.ffn.{index}.{suffix}"))
}
fn value(index: usize, suffix: &str) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!("value.fast_h3.ffn.{index}.{suffix}"))
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("FastH3 FFN: {}", reason.into()),
    }
}
