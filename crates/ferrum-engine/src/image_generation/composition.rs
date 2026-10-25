//! Backend-independent admission limits and real-graph command accounting.
use super::*;
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
use anyhow::Context;
use anyhow::{ensure, Result};
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
use ferrum_models::qwen3_vl_text::Qwen3VLTextImageSpan;
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
use ferrum_models::qwen3_vl_vision::{
    Qwen3VLVisionArchitecture, Qwen3VLVisionConfig, Qwen3VLVisionFamily, Qwen3VLVisionGrid,
};
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
use ferrum_models::qwen_image21::{QwenImage21VaeEncoderConfig, QwenImage21VaeEncoderFamily};
use std::num::NonZeroU64;

/// Request limits without an independent device budget. The factory supplies
/// the actual runtime ceiling to every component plan after context creation.
#[derive(Debug, Clone)]
pub struct ImagePipelineLimits {
    pub maximum_user_tokens: usize,
    pub maximum_width: u32,
    pub maximum_height: u32,
    pub maximum_batch_size: usize,
    pub decoder_batch_policy: super::ImageDecoderBatchPolicy,
    pub edit_prefix_policy: super::ImageEditPrefixPolicy,
    pub plan_cache: super::ImagePlanCachePolicy,
    pub residency: super::ImageResidencyPolicy,
    pub dynamic_residency: super::ImageDynamicResidencyPolicy,
    pub static_initialization: StaticInitializationPolicy,
    pub executor: ferrum_execution::TensorExecutorConfig,
}
impl ImagePipelineLimits {
    pub fn validate(&self) -> Result<()> {
        self.plan_cache.validate()?;
        self.dynamic_residency.validate(self.residency)?;
        ensure!(
            self.maximum_width > 0 && self.maximum_height > 0,
            "image dimension limits must be positive"
        );
        ensure!(
            (1..=8192).contains(&self.maximum_user_tokens),
            "user-token limit must be 1..=8192"
        );
        ensure!(
            (1..=STRUCTURED_ATTENTION_BF16_MAX_BATCH).contains(&self.maximum_batch_size),
            "image maximum_batch_size exceeds supported attention capacity"
        );
        Ok(())
    }
    pub fn with_device_capacity(self, capacity: NonZeroU64) -> ImagePipelineConfig {
        ImagePipelineConfig {
            device_capacity_bytes: capacity.get(),
            maximum_user_tokens: self.maximum_user_tokens,
            maximum_width: self.maximum_width,
            maximum_height: self.maximum_height,
            maximum_batch_size: self.maximum_batch_size,
            decoder_batch_policy: self.decoder_batch_policy,
            edit_prefix_policy: self.edit_prefix_policy,
            plan_cache: self.plan_cache,
            residency: self.residency,
            dynamic_residency: self.dynamic_residency,
            static_initialization: self.static_initialization,
            executor: self.executor,
        }
    }
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
pub(super) fn component_command_count(family: &PreparedModelFamily) -> Result<usize> {
    family
        .program()
        .blocks()
        .iter()
        .try_fold(family.program().inputs().len(), |n, block| {
            n.checked_add(block.nodes.len())
        })
        .context("component command count overflows")
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
pub(super) fn validate_family_command_limits(
    families: &[PreparedModelFamily],
    maximum: usize,
) -> Result<()> {
    for family in families {
        let commands = component_command_count(family)?;
        ensure!(commands <= maximum,
            "{} requires at least {commands} compute/upload commands, exceeding configured {maximum}",
            family.family_id());
    }
    Ok(())
}

/// Small metadata-only representatives of every additional editing topology.
/// Tensor contents and request dimensions cannot reduce these command counts.
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
pub(super) fn editing_command_families(
    text: &Qwen3VLTextConfig,
    dit: &QwenImage21TransformerConfig,
    vision: &Qwen3VLVisionArchitecture,
    encoder: &QwenImage21VaeEncoderConfig,
    profiles: &QwenImage21Profiles,
    editing_profiles: &super::QwenImage21EditProfiles,
) -> Result<[PreparedModelFamily; 4]> {
    let merge = vision.merge_size();
    let visual = Qwen3VLVisionConfig::new(
        vision.clone(),
        vec![Qwen3VLVisionGrid::new(1, merge, merge)?],
    )?;
    let text = text
        .clone()
        .with_image_spans(&[Qwen3VLTextImageSpan::new(0, merge, merge, merge)?])?;
    let dit = QwenImage21TransformerConfig::new(
        dit.transformer().clone(),
        QwenImage21T2iLayout::new(3, 2, 2, vec![3])?,
    )?
    .with_weight_encodings(dit.weight_encodings().clone())?
    .with_reference_image(1, 1, 2, 2)?;
    Ok([
        TypedFamilyRegistration::new(Qwen3VLVisionFamily::new()?)
            .prepare_with_profile(&serde_json::to_value(visual)?, &editing_profiles.vision)?,
        TypedFamilyRegistration::new(QwenImage21VaeEncoderFamily::new()?)
            .prepare_with_profile(&serde_json::to_value(encoder)?, &editing_profiles.encoder)?,
        TypedFamilyRegistration::new(Qwen3VLTextEncoderFamily::new()?)
            .prepare_with_profile(&serde_json::to_value(text)?, &profiles.text_encoder)?,
        TypedFamilyRegistration::new(QwenImage21TransformerFamily::new()?)
            .prepare_with_profile(&serde_json::to_value(dit)?, &profiles.denoiser)?,
    ])
}
