//! Complete block-0 arithmetic with explicit caller-supplied conditioning.
//!
//! The six inputs are hidden states, two raw scales, two raw gates and RoPE
//! frequencies. The caller selects the timestep/t0 conditioning rows before
//! invocation. This family does not generate conditioning, frequencies or the
//! rest of the transformer. Every node boundary is exposed for diagnostics.
//!
//! The scale-only modulation, tanh gates and two residual paths follow
//! `qwen_image_21_transformer.py` lines 649–723 at vLLM-Omni revision
//! `c0551d22674d0db81305c800b66ad83035fdc980`. Attention and MLP reuse the
//! existing partial families; neither has an affine LayerNorm or a shift input.

mod config;
mod program;

pub use config::{QwenImage21Block0Broadcast, QwenImage21Block0Config};

use super::{
    QwenImage21AttentionConfig, QwenImage21Block0AttentionFamily as Attention,
    QwenImage21Block0MlpFamily as Mlp, QwenImage21Config,
};
use ferrum_runtime::*;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
pub struct QwenImage21Block0Family {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
}

impl QwenImage21Block0Family {
    pub const FAMILY_ID: &'static str = "family.qwen_image21.block0";
    pub const REFERENCE_PROFILE_ID: &'static str = "qwen_image21.block0.reference.bf16_f32.v1";
    pub const CUDA_PROFILE_ID: &'static str = "qwen_image21.block0.cuda.bf16_f32.v1";
    /// Explicit experimental vendor-fused attention; other block operations
    /// keep the existing CUDA numerical choices.
    pub const CUDA_FUSED_PROFILE_ID: &'static str =
        "qwen_image21.block0.cuda.fused_attention.experimental.v1";
    pub const INPUT_ID: &'static str = "value.qwen_image21.block0.x";
    pub const SCALE1_ID: &'static str = "value.qwen_image21.block0.scale1";
    pub const GATE1_ID: &'static str = "value.qwen_image21.block0.gate1";
    pub const SCALE2_ID: &'static str = "value.qwen_image21.block0.scale2";
    pub const GATE2_ID: &'static str = "value.qwen_image21.block0.gate2";
    pub const FREQUENCIES_ID: &'static str = "value.qwen_image21.block0.frequencies";
    pub const NORMALIZED1_ID: &'static str = "value.qwen_image21.block0.normalized1";
    pub const MODULATED1_ID: &'static str = "value.qwen_image21.block0.modulated1";
    pub const TANH_GATE1_ID: &'static str = "value.qwen_image21.block0.tanh_gate1";
    pub const RESIDUAL1_ID: &'static str = "value.qwen_image21.block0.residual1";
    pub const NORMALIZED2_ID: &'static str = "value.qwen_image21.block0.normalized2";
    pub const MODULATED2_ID: &'static str = "value.qwen_image21.block0.modulated2";
    pub const TANH_GATE2_ID: &'static str = "value.qwen_image21.block0.tanh_gate2";
    pub const OUTPUT_ID: &'static str = "value.qwen_image21.block0.output";
    pub const INPUT_IDS: [&'static str; 6] = [
        Self::INPUT_ID,
        Self::SCALE1_ID,
        Self::GATE1_ID,
        Self::SCALE2_ID,
        Self::GATE2_ID,
        Self::FREQUENCIES_ID,
    ];
    /// Includes the thirteen existing attention/MLP boundaries without renaming
    /// their values. New node identities belong to this full-block family.
    pub const OUTPUT_IDS: [&'static str; 21] = [
        Self::NORMALIZED1_ID,
        Self::MODULATED1_ID,
        Self::TANH_GATE1_ID,
        Attention::QUERY_ID,
        Attention::KEY_ID,
        Attention::QUERY_NORMALIZED_ID,
        Attention::KEY_NORMALIZED_ID,
        Attention::QUERY_ROTATED_ID,
        Attention::KEY_ROTATED_ID,
        Attention::VALUE_ID,
        Attention::ATTENTION_ID,
        Attention::OUTPUT_ID,
        Self::RESIDUAL1_ID,
        Self::NORMALIZED2_ID,
        Self::MODULATED2_ID,
        Self::TANH_GATE2_ID,
        Mlp::GATE_ID,
        Mlp::UP_ID,
        Mlp::ACTIVATION_ID,
        Mlp::OUTPUT_ID,
        Self::OUTPUT_ID,
    ];
    pub const WEIGHT_NAMES: [&'static str; 9] = [
        Attention::WEIGHT_NAMES[0],
        Attention::WEIGHT_NAMES[1],
        Attention::WEIGHT_NAMES[2],
        Attention::WEIGHT_NAMES[3],
        Attention::WEIGHT_NAMES[4],
        Attention::WEIGHT_NAMES[5],
        Mlp::WEIGHT_NAMES[0],
        Mlp::WEIGHT_NAMES[1],
        Mlp::WEIGHT_NAMES[2],
    ];

    pub fn new() -> Result<Self, VNextError> {
        Ok(Self {
            id: ModelFamilyId::new(Self::FAMILY_ID)?,
            metadata: ExternalModelMetadataId::new("metadata.qwen_image21.block0")?,
        })
    }
}

impl ModelFamilyProvider for QwenImage21Block0Family {
    type Config = QwenImage21Block0Config;

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
        if serde_json::to_value(config).map_err(|e| invalid("config", e.to_string()))? != *raw {
            return Err(invalid(
                "config",
                "block 0 requires canonical checked layout and conditioning config",
            ));
        }
        config.validate()
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
        config.validate()?;
        let ModelWeights::Stored(mut attention) = Attention::new()?.weights(config.attention())?
        else {
            return Err(invalid(
                "weights",
                "attention branch must have Stored weights",
            ));
        };
        let ModelWeights::Stored(mlp) = Mlp::new()?.weights(config.transformer())? else {
            return Err(invalid("weights", "MLP branch must have Stored weights"));
        };
        if attention.format_id != mlp.format_id || attention.version != mlp.version {
            return Err(invalid(
                "weights",
                "branch weight formats or versions disagree",
            ));
        }
        attention.components.extend(mlp.components);
        attention.tensors.extend(mlp.tensors);
        attention.layout_id = WeightLayoutId::new("weight-layout.qwen_image21.block0.bf16")?;
        attention.validate(&self.id)?;
        Ok(ModelWeights::Stored(attention))
    }

    fn numerical_profiles(
        &self,
        config: &Self::Config,
    ) -> Result<FamilyNumericalProfiles, VNextError> {
        config.validate()?;
        let attention = Attention::new()?.numerical_profiles(config.attention())?;
        let mlp = Mlp::new()?.numerical_profiles(config.transformer())?;
        let mut profiles = Vec::new();
        for (name, attention_name, mlp_name) in profile_sources() {
            if name == Self::CUDA_FUSED_PROFILE_ID && config.transformer().head_dim() != 128 {
                continue;
            }
            let mut profile = attention.resolve(&profile_id(attention_name)?)?.clone();
            let mlp = mlp.resolve(&profile_id(mlp_name)?)?;
            if !profile.states.is_empty()
                || !profile.kv_storage.is_empty()
                || !mlp.states.is_empty()
                || !mlp.kv_storage.is_empty()
            {
                return Err(invalid(
                    "profile",
                    "block branches must be stateless tensor components",
                ));
            }
            let mut operations: BTreeMap<_, _> = profile
                .operations
                .into_iter()
                .map(|op| (op.operation_id.clone(), op))
                .collect();
            for operation in &mlp.operations {
                match operations.get(&operation.operation_id) {
                    Some(existing) if existing != operation => {
                        return Err(invalid(
                            "profile",
                            "branches disagree on a shared operation",
                        ))
                    }
                    Some(_) => {}
                    None => {
                        operations.insert(operation.operation_id.clone(), operation.clone());
                    }
                }
            }
            for name in [
                LAYER_NORM_BF16_OPERATION_ID,
                MODULATION_SCALE_BF16_OPERATION_ID,
                TANH_BF16_OPERATION_ID,
                GATED_RESIDUAL_BF16_OPERATION_ID,
            ] {
                let operation = NumericalOperationContract {
                    operation_id: OperationId::new(name)?,
                    version: ContractVersion::new(1, 0),
                    multiplication_type: Some(ElementType::F32),
                    accumulation_type: Some(ElementType::F32),
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                };
                if operations
                    .insert(operation.operation_id.clone(), operation)
                    .is_some()
                {
                    return Err(invalid(
                        "profile",
                        "branch unexpectedly owns a block wrapper operation",
                    ));
                }
            }
            profile.id = profile_id(name)?;
            profile.family_id = self.id.clone();
            profile.primary_activation = ProgramValueId::new(Self::OUTPUT_ID)?;
            profile.operations = operations.into_values().collect();
            for (id, dtype) in &mlp.boundaries {
                if profile.boundaries.insert(id.clone(), *dtype).is_some() {
                    return Err(invalid(
                        "profile",
                        "branches have conflicting boundary identities",
                    ));
                }
            }
            for name in [
                Self::NORMALIZED1_ID,
                Self::MODULATED1_ID,
                Self::TANH_GATE1_ID,
                Self::RESIDUAL1_ID,
                Self::NORMALIZED2_ID,
                Self::MODULATED2_ID,
                Self::TANH_GATE2_ID,
                Self::OUTPUT_ID,
            ] {
                profile
                    .boundaries
                    .insert(ProgramValueId::new(name)?, ElementType::Bf16);
            }
            profiles.push(profile);
        }
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            profiles,
            vec![profile_id(Self::REFERENCE_PROFILE_ID)?],
        )
    }

    fn semantic_program(
        &self,
        config: &Self::Config,
        profile: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        config.validate()?;
        if self.numerical_profiles(config)?.resolve(&profile.id)? != profile {
            return Err(invalid(
                "profile",
                "full-block profile differs from its declared arithmetic",
            ));
        }
        program::build(self, config, profile)
    }

    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

fn profile_sources() -> [(&'static str, &'static str, &'static str); 3] {
    [
        (
            QwenImage21Block0Family::REFERENCE_PROFILE_ID,
            Attention::REFERENCE_PROFILE_ID,
            Mlp::REFERENCE_PROFILE_ID,
        ),
        (
            QwenImage21Block0Family::CUDA_PROFILE_ID,
            Attention::CUDA_PROFILE_ID,
            Mlp::CUDA_PROFILE_ID,
        ),
        (
            QwenImage21Block0Family::CUDA_FUSED_PROFILE_ID,
            Attention::CUDA_FUSED_PROFILE_ID,
            Mlp::CUDA_PROFILE_ID,
        ),
    ]
}

fn profile_id(name: &str) -> Result<NumericalProfileId, VNextError> {
    NumericalProfileId::new(name).map_err(|e| invalid("profile", e))
}

fn invalid(field: &str, reason: impl Into<String>) -> VNextError {
    VNextError::InvalidModelConfig {
        family_id: QwenImage21Block0Family::FAMILY_ID.into(),
        field: field.into(),
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests;
