//! Complete single-frame decoder network: post-quant convolution, both mid
//! residuals and spatial attention, every residual up block, and raw RGBA head.
//!
//! Source: vllm-omni c0551d22674d0db81305c800b66ad83035fdc980,
//! `autoencoder_kl_qwenimage21.py`, DupUp3D/Resample/ResidualUpBlock/Decoder3d
//! and `_decode`. The single-frame first chunk never executes time_conv.
//! Ordinary upsampler convolutions retain implicit symmetric padding; causal
//! convolutions retain explicit zero padding. No video/cache/tiling semantics
//! are inferred. Independent provider profiles do not promise Torch bit parity.
//!
//! The input is ALREADY denormalized BF16 NCHW. The pipeline must apply the
//! source's B(B(latent * B(std)) + B(mean)) before this family (the configured
//! mean/std first enter FP32 tensors, then BF16). The output is raw
//! BF16 RGBA; `_decode`'s clamp[-1,1] and later pixel/PNG conversion belong to the
//! pipeline adapter, not to this graph. FP32-only denormalization is insufficient.

mod config;
mod graph;
mod weights;
pub use config::QwenImage21VaeDecoderConfig;
use ferrum_runtime::*;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
pub use weights::QwenImage21VaeDecoderWeight;

#[derive(Debug, Clone)]
pub struct QwenImage21VaeDecoderFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
}
impl QwenImage21VaeDecoderFamily {
    pub const FAMILY_ID: &'static str = "family.qwen_image21.vae.decoder";
    pub const REFERENCE_PROFILE_ID: &'static str = "qwen_image21.vae.decoder.reference.bf16_f32.v1";
    pub const CUDA_PROFILE_ID: &'static str = "qwen_image21.vae.decoder.cuda.bf16_f32.v1";
    pub const METAL_PROFILE_ID: &'static str = "qwen_image21.vae.decoder.metal.bf16_f32.v1";
    pub const INPUT_ID: &'static str = "value.qwen_image21.vae.decoder.denormalized_latent";
    pub const OUTPUT_ID: &'static str = "value.qwen_image21.vae.decoder.decoder.conv_out";
    pub fn new() -> Result<Self, VNextError> {
        Ok(Self {
            id: ModelFamilyId::new(Self::FAMILY_ID)?,
            metadata: ExternalModelMetadataId::new("metadata.qwen_image21.vae.decoder")?,
        })
    }
}
impl ModelFamilyProvider for QwenImage21VaeDecoderFamily {
    type Config = QwenImage21VaeDecoderConfig;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([self.metadata.clone()])
    }
    fn validate_config_identity(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<(), VNextError> {
        config.validate()?;
        if serde_json::to_value(config).map_err(|e| invalid("config", e.to_string()))? != *raw {
            return Err(invalid(
                "config",
                "requires canonical single-frame decoder configuration",
            ));
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(self.metadata.clone())
    }
    fn parse_config(&self, raw: &Value) -> Result<Self::Config, VNextError> {
        let config =
            serde_json::from_value(raw.clone()).map_err(|e| invalid("config", e.to_string()))?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }
    fn weights(&self, config: &Self::Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::Stored(weights::schema(
            &config.expected_weights()?,
            &self.id,
        )?))
    }
    fn numerical_profiles(
        &self,
        config: &Self::Config,
    ) -> Result<FamilyNumericalProfiles, VNextError> {
        let graph = graph::build(config)?;
        let operations = graph
            .nodes
            .iter()
            .map(|n| n.operation_id.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|operation_id| {
                let transport = operation_id.as_str() == SPATIAL_UPSAMPLE_GATHER_BF16_OPERATION_ID;
                let add = operation_id.as_str() == PLAIN_RESIDUAL_ADD_NCHW_BF16_OPERATION_ID;
                NumericalOperationContract {
                    operation_id,
                    version: ContractVersion::new(1, 0),
                    multiplication_type: (!transport && !add).then_some(ElementType::F32),
                    accumulation_type: (!transport).then_some(ElementType::F32),
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                }
            })
            .collect::<Vec<_>>();
        let boundaries = graph
            .specs
            .into_keys()
            .map(|id| (id, ElementType::Bf16))
            .collect::<BTreeMap<_, _>>();
        let profiles = [
            Self::REFERENCE_PROFILE_ID,
            Self::CUDA_PROFILE_ID,
            Self::METAL_PROFILE_ID,
        ]
        .into_iter()
        .map(|id| {
            Ok(NumericalExecutionProfile {
                id: NumericalProfileId::new(id).map_err(|e| invalid("profile", e))?,
                family_id: self.id.clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: ProgramValueId::new(Self::OUTPUT_ID)?,
                boundaries: boundaries.clone(),
                operations: operations.clone(),
                kv_storage: vec![],
                states: vec![],
            })
        })
        .collect::<Result<Vec<_>, VNextError>>()?;
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            profiles,
            vec![NumericalProfileId::new(Self::REFERENCE_PROFILE_ID)
                .map_err(|e| invalid("profile", e))?],
        )
    }
    fn semantic_program(
        &self,
        config: &Self::Config,
        profile: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        if self.numerical_profiles(config)?.resolve(&profile.id)? != profile {
            return Err(invalid("profile", "undeclared decoder profile"));
        }
        let graph = graph::build(config)?;
        let references = graph
            .weights
            .iter()
            .map(|w| {
                Ok(WeightReference {
                    weight_id: w.weight_id()?,
                    value_id: w.value_id()?,
                    tensor: spec(w.dimensions()),
                })
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
        ModelProgram::new(
            self.id.clone(),
            vec![ProgramValueId::new(Self::INPUT_ID)?],
            vec![ProgramBlock {
                id: "block.qwen_image21.vae.decoder".into(),
                nodes: graph.nodes,
            }],
            vec![],
            references,
            vec![ProgramValueId::new(Self::OUTPUT_ID)?],
        )?
        .with_input_tensor_specs(config.input_tensor_specs()?)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn spec(dimensions: &[u64]) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions: dimensions.to_vec(),
        element_type: ElementType::Bf16,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
fn tensor(dimensions: &[u64]) -> Result<ResolvedTensorSpec, VNextError> {
    ResolvedTensorSpec::new(
        dimensions.to_vec(),
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
}
fn invalid(field: &str, reason: impl Into<String>) -> VNextError {
    VNextError::InvalidModelConfig {
        family_id: QwenImage21VaeDecoderFamily::FAMILY_ID.into(),
        field: field.into(),
        reason: reason.into(),
    }
}
#[cfg(test)]
mod tests;
