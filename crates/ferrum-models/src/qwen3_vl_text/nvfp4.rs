//! Serialized NVFP4 projections over the existing BF16 Qwen3-VL text graph.
//!
//! The checkpoint explicitly declares a retained decoder prefix. Inputs remain
//! caller-provided embeddings and rotary tensors; output is pre-final-norm.
//! Activation quantization uses unit global SF. The numerical profile reuses
//! Ferrum's BF16 RMS/RoPE/online attention and staged SiLU contracts; it does
//! not assert end-to-end equivalence to another attention implementation.

use super::{Qwen3VLTextConfig, Qwen3VLTextEncoderFamily as DenseFamily};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

mod config;
mod program;
mod source;
pub use config::{Qwen3VLTextNvfp4Attention, Qwen3VLTextNvfp4Config};
pub use source::{Qwen3VLTextNvfp4Checkpoint, Qwen3VLTextNvfp4Limits, Qwen3VLTextNvfp4Source};

const SCALAR_PROFILE: &str = "nvfp4.serialized.f32_global_reciprocal_f64_to_f32.unit_activation.v1";
const UNIT_NAME: &str = "derived.qwen3_vl.nvfp4.unit_global_scale";

fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidModelConfig {
        family_id: Qwen3VLTextNvfp4Family::FAMILY_ID.into(),
        field: "nvfp4".into(),
        reason: reason.into(),
    }
}

#[derive(Clone, Debug)]
pub struct Qwen3VLTextNvfp4Family {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
}
impl Qwen3VLTextNvfp4Family {
    pub const FAMILY_ID: &'static str = "family.qwen3_vl.text_encoder.nvfp4";
    pub const CUDA_PROFILE_ID: &'static str = "qwen3vl.text.cuda.nvfp4_unit.bf16_f32.v1";
    pub const CUDA_FUSED_PROFILE_ID: &'static str =
        "qwen3vl.text.cuda.nvfp4_unit.fused_causal_gqa.bf16_f32.v1";
    pub fn new() -> Result<Self, VNextError> {
        Ok(Self {
            id: ModelFamilyId::new(Self::FAMILY_ID)?,
            metadata: ExternalModelMetadataId::new("metadata.qwen3_vl.text_encoder.nvfp4")?,
        })
    }
}
impl ModelFamilyProvider for Qwen3VLTextNvfp4Family {
    type Config = Qwen3VLTextNvfp4Config;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([self.metadata.clone()])
    }
    fn validate_config_identity(&self, raw: &Value, c: &Self::Config) -> Result<(), VNextError> {
        c.validate()?;
        if serde_json::to_value(c).map_err(|e| invalid(e.to_string()))? != *raw {
            return Err(invalid("requires exact canonical checked configuration"));
        }
        Ok(())
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
        c.validate()?;
        Ok(ModelWeights::Stored(c.schema()?))
    }
    fn numerical_profiles(&self, c: &Self::Config) -> Result<FamilyNumericalProfiles, VNextError> {
        let graph = program::build(c, &self.id)?;
        let base = DenseFamily::new()?.numerical_profiles(c.text())?;
        let base =
            base.resolve(&NumericalProfileId::new(DenseFamily::CUDA_PROFILE_ID).map_err(invalid)?)?;
        let mut operations: Vec<_> = base
            .operations
            .iter()
            .filter(|op| op.operation_id.as_str() != DENSE_LINEAR_BF16_OPERATION_ID)
            .cloned()
            .collect();
        if c.attention == Qwen3VLTextNvfp4Attention::VendorFusedCausal {
            for op in &mut operations {
                if op.operation_id.as_str() == STRUCTURED_GQA_ATTENTION_BF16_OPERATION_ID {
                    op.operation_id =
                        OperationId::new(STRUCTURED_GQA_ATTENTION_FUSED_BF16_OPERATION_ID)?;
                    op.multiplication_type = None;
                    op.accumulation_type = None;
                    op.staged_arithmetic = None;
                    op.composite_arithmetic = None;
                }
            }
        }
        for name in [
            NVFP4_QUANTIZE_BF16_OPERATION_ID,
            NVFP4_MATMUL_BF16_OPERATION_ID,
        ] {
            operations.push(NumericalOperationContract {
                operation_id: OperationId::new(name)?,
                version: ContractVersion::new(1, 0),
                multiplication_type: None,
                accumulation_type: None,
                staged_arithmetic: None,
                composite_arithmetic: None,
            });
        }
        let boundaries = graph
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .flat_map(|n| {
                let dtype = if n.operation_id.as_str() == NVFP4_QUANTIZE_BF16_OPERATION_ID {
                    ElementType::U8
                } else {
                    ElementType::Bf16
                };
                n.outputs.iter().cloned().map(move |id| (id, dtype))
            })
            .collect();
        let id = NumericalProfileId::new(c.numerical_profile_id()).map_err(invalid)?;
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: id.clone(),
                family_id: self.id.clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: ProgramValueId::new(DenseFamily::OUTPUT_ID)?,
                boundaries,
                states: vec![],
                kv_storage: vec![],
                operations,
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
            return Err(invalid(
                "numerical profile differs from declared arithmetic",
            ));
        }
        program::build(c, &self.id)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

#[cfg(test)]
mod tests;
