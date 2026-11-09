//! Typed acceptance limits for the original HF image processor. These are
//! caps, not reservations; actual image payloads use the pipeline Host account.
use super::*;
use ferrum_models::{
    qwen3_vl_vision::hf::Qwen3VLVisionHfProcessorLimits,
    qwen_image21::image_input::ImageInputLimits,
};

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PH3ImagePromptExpansionVisionLimits {
    pub maximum_encoded_bytes: usize,
    pub maximum_pixels: usize,
    pub maximum_width: u32,
    pub maximum_height: u32,
    pub decoder_workspace_bytes: usize,
    pub max_patch_rows: u64,
    pub max_patch_bytes: usize,
    pub max_metadata_bytes: usize,
    pub max_plan_nodes: usize,
    pub max_stored_bytes: u64,
}
impl Default for PH3ImagePromptExpansionVisionLimits {
    fn default() -> Self {
        let image = ImageInputLimits::default();
        let family = ferrum_engine::image_generation::p_h3::PH3ImageAssetLimits::default().vision;
        Self {
            maximum_encoded_bytes: image.maximum_encoded_bytes(),
            maximum_pixels: image.maximum_pixels(),
            maximum_width: image.maximum_width(),
            maximum_height: image.maximum_height(),
            decoder_workspace_bytes: image.decoder_workspace_bytes(),
            max_patch_rows: 1_000_000,
            max_patch_bytes: 512 << 20,
            max_metadata_bytes: family.max_metadata_bytes,
            max_plan_nodes: family.max_plan_nodes,
            max_stored_bytes: family.max_stored_bytes,
        }
    }
}
impl PH3ImagePromptExpansionVisionLimits {
    pub(super) fn resolved(
        self,
        max_processor_bytes: usize,
    ) -> Result<(Qwen3VLVisionHfProcessorLimits, Qwen3VLVisionHfLimits)> {
        ensure!(
            max_processor_bytes > 0
                && max_processor_bytes <= isize::MAX as usize
                && self.max_patch_rows > 0
                && self.max_patch_bytes > 0
                && self.max_patch_bytes <= isize::MAX as usize
                && self.max_metadata_bytes > 0
                && self.max_metadata_bytes <= isize::MAX as usize
                && self.max_plan_nodes > 0
                && self.max_stored_bytes > 0
                && self.max_stored_bytes <= isize::MAX as u64,
            "prompt expansion vision limits must be positive and addressable"
        );
        let image = ImageInputLimits::new(
            self.maximum_encoded_bytes,
            self.maximum_pixels,
            self.maximum_width,
            self.maximum_height,
            self.decoder_workspace_bytes,
        )?;
        Ok((
            Qwen3VLVisionHfProcessorLimits {
                max_processor_bytes,
                image,
                max_patch_rows: self.max_patch_rows,
                max_patch_bytes: self.max_patch_bytes,
            },
            Qwen3VLVisionHfLimits {
                max_metadata_bytes: self.max_metadata_bytes,
                max_plan_nodes: self.max_plan_nodes,
                max_stored_bytes: self.max_stored_bytes,
            },
        ))
    }
}
impl PH3ImagePromptExpansionEnabledConfig {
    pub(in crate::commands::image_product::p_h3) fn prepare_processor(
        &self,
    ) -> Result<(Qwen3VLVisionHfProcessor, Qwen3VLVisionHfLimits)> {
        let (limits, family) = self
            .vision_limits
            .resolved(self.metadata_limits.max_processor_bytes)?;
        let bytes = super::super::super::source::read_bounded(
            &self.processor_config,
            limits.max_processor_bytes,
        )
        .with_context(|| {
            format!(
                "read original prompt expansion processor {}",
                self.processor_config.display()
            )
        })?;
        let processor = Qwen3VLVisionHfProcessor::from_source_bytes(&bytes, limits)?;
        Ok((processor, family))
    }
}
