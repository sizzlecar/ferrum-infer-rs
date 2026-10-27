//! Metadata bounds for overlapping completed output and the next execution.
use super::*;
use anyhow::Context;
use ferrum_models::{normal_noise::NormalNoiseConfig, qwen3_vl_vision::Qwen3VLVisionFamily};
use ferrum_runtime::{ResolvedTensorLayout, ResolvedTensorSpec};

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ImageOutputRequirements {
    pub batch_size: usize,
    pub maximum_width: u32,
    pub maximum_height: u32,
    /// Completed decoder backing retained by all images of one batch.
    pub raw_batch_bytes: usize,
    /// Conservative declared upper bound on any stage's simultaneous readbacks,
    /// including edit outputs; not a measured or exact executable peak.
    pub execution_output_bytes: usize,
}

impl QwenImage21Assets {
    /// No compilation or payload allocation. Bounds use the decoder geometry,
    /// tokenizer admission rule and processor metadata used by real requests.
    /// Processor ceilings may exceed the reachable resized reference geometry.
    /// Live preparation still validates and reserves its exact compiled ports.
    pub fn output_requirements(
        &self,
        limits: &ImagePipelineLimits,
        host: &HostMemoryAccount,
    ) -> anyhow::Result<ImageOutputRequirements> {
        limits.validate()?;
        let config = limits
            .clone()
            .with_device_capacity(std::num::NonZeroU64::MIN);
        let request = ImageGenerationRequest {
            prompt: String::new(),
            width: limits.maximum_width,
            height: limits.maximum_height,
            seed: 0,
            n: 1,
        };
        let (height, width) = request.latent_dimensions(&config, self.spatial_scale()?)?;
        let batch = limits.maximum_batch_size;
        let physical = |tensor: &ResolvedTensorSpec| -> anyhow::Result<usize> {
            Ok(host.backing_bytes(usize::try_from(tensor.minimum_storage_bytes()?)?)?)
        };
        let decoder =
            self.decoder_batch(limits.decoder_batch_policy.batch_size(batch), height, width)?;
        let raw = super::host_budget::decoder_output_peak(
            physical(&decoder.output_tensor()?)?,
            batch,
            limits.decoder_batch_policy,
        )?;
        let latent = QwenImage21LatentShape::new(batch, height as usize, width as usize)?;
        let mut peak = raw
            .max(physical(&latent.tensor(QwenImage21LatentLayout::Packed)?)?)
            .max(physical(
                &NormalNoiseConfig::new(64, height.into(), width.into())?.output_tensor()?,
            )?);
        let mut visual_tokens = 0;
        if let Some(edit) = &self.editing {
            let pixels = edit.processor.maximum_condition_pixels();
            let factor = edit
                .processor
                .patch_size()
                .checked_mul(edit.processor.merge_size())
                .context("vision merge factor overflows")?;
            visual_tokens = pixels
                / factor
                    .checked_mul(factor)
                    .context("vision patch area overflows")?;
            let merged = ResolvedTensorSpec::new(
                vec![u64::try_from(visual_tokens)?, edit.vision.output_size()],
                ElementType::Bf16,
                ResolvedTensorLayout::Contiguous,
            )?;
            // Main merger plus every DeepStack readback remains until the text
            // bridge has copied every reference in the request batch.
            peak = peak.max(
                physical(&merged)?
                    .checked_mul(1 + Qwen3VLVisionFamily::DEEPSTACK_IDS.len())
                    .and_then(|n| n.checked_mul(batch))
                    .context("vision readback bound overflows")?,
            );
            let scale = usize::try_from(edit.encoder.spatial_scale())?;
            let reference = ResolvedTensorSpec::new(
                vec![
                    1,
                    edit.encoder.latent_channels(),
                    u64::try_from(pixels / (scale * scale))?,
                ],
                edit.encoder.output_tensor()?.element_type(),
                ResolvedTensorLayout::Contiguous,
            )?;
            peak = peak.max(physical(&reference)?);
        }
        let sequence = self
            .tokenizer
            .maximum_sequence_length(limits.maximum_user_tokens, visual_tokens)?;
        let hidden = ResolvedTensorSpec::new(
            vec![
                u64::try_from(batch)?,
                u64::try_from(sequence)?,
                self.text.hidden_size(),
            ],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        peak = peak.max(physical(&hidden)?);
        Ok(ImageOutputRequirements {
            batch_size: batch,
            maximum_width: limits.maximum_width,
            maximum_height: limits.maximum_height,
            raw_batch_bytes: raw,
            execution_output_bytes: peak,
        })
    }
}
