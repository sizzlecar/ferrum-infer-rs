//! Readback peaks from typed model geometry, before compilation or decoding.
use super::*;
use crate::image_generation::host_budget::decoder_output_peak;
use anyhow::{ensure, Context};
use ferrum_models::{
    normal_noise::NormalNoiseConfig,
    qwen3_vl_text::{Qwen3VLTextConfig, Qwen3VLTextEncoderFamily},
    qwen3_vl_vision::{Qwen3VLVisionConfig, Qwen3VLVisionFamily, Qwen3VLVisionGrid},
    qwen_image21::{
        pipeline_adapters::{QwenImage21LatentLayout, QwenImage21LatentShape},
        t2i_prompt::QwenImage21EditImageGrid,
    },
};
use ferrum_runtime::{ElementType, ProgramValueId, ResolvedTensorLayout, ResolvedTensorSpec};

fn physical(host: &HostMemoryAccount, tensor: &ResolvedTensorSpec) -> anyhow::Result<usize> {
    Ok(host.backing_bytes(usize::try_from(tensor.minimum_storage_bytes()?)?)?)
}

impl<R: DeviceRuntime> QwenImage21Pipeline<R> {
    pub(super) fn service_execution_output_bytes(
        &self,
        requests: &[ImageRequestRef<'_>],
    ) -> anyhow::Result<usize> {
        ensure!(
            !requests.is_empty() && requests.len() <= self.execution.config.maximum_batch_size,
            "output geometry requires a supported nonempty batch"
        );
        let first = requests[0].generation();
        let (height, width) = first.latent_dimensions(
            &self.execution.config,
            self.execution.sources.assets.spatial_scale()?,
        )?;
        let edit = matches!(requests[0], ImageRequestRef::Edit(_));
        for request in requests {
            let generation = request.generation();
            ensure!(
                generation.width == first.width
                    && generation.height == first.height
                    && matches!(request, ImageRequestRef::Edit(_)) == edit,
                "output geometry requires compatible request types and dimensions"
            );
            generation.latent_dimensions(
                &self.execution.config,
                self.execution.sources.assets.spatial_scale()?,
            )?;
        }
        let batch = requests.len();
        let prompts: Vec<_> = requests
            .iter()
            .map(|request| request.generation().prompt.as_str())
            .collect();
        let mut edit_output = 0usize;
        let text = if edit {
            let assets = self
                .execution
                .sources
                .assets
                .editing
                .as_ref()
                .context("editing assets are missing")?;
            let mut geometry: Option<Vec<(u32, u32)>> = None;
            for request in requests {
                let ImageRequestRef::Edit(request) = request else {
                    unreachable!()
                };
                ensure!(
                    (1..=self.maximum_reference_images()).contains(&request.image_count()),
                    "reference image count exceeds the model capability"
                );
                let current = request
                    .images()
                    .map(|image| {
                        let (w, h) =
                            png_dimensions_hint(image.bytes(), ImageInputLimits::default())
                                .context("invalid bounded reference PNG header")?;
                        // Validate every reference without allocating its backing.
                        assets
                            .processor
                            .prepared_backing_bytes(&self.execution.host, w, h)?;
                        Ok(QwenImage21ImageProcessor::condition_dimensions(w, h)?)
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?;
                ensure!(
                    geometry
                        .as_ref()
                        .is_none_or(|previous| previous == &current),
                    "ordered edit reference geometries differ"
                );
                geometry = Some(current);
            }
            let geometry = geometry.context("missing edit reference geometry")?;
            let patch = u32::try_from(assets.processor.patch_size())?;
            let mut grids = Vec::with_capacity(geometry.len());
            let mut vision_outputs = 0usize;
            let mut encoder_output = 0usize;
            for (w, h) in geometry {
                let grid = Qwen3VLVisionGrid::new(1, u64::from(h / patch), u64::from(w / patch))?;
                let vision = Qwen3VLVisionConfig::new(assets.vision.clone(), vec![grid])?;
                let merged = ResolvedTensorSpec::new(
                    vec![vision.merged_rows()?, assets.vision.output_size()],
                    ElementType::Bf16,
                    ResolvedTensorLayout::Contiguous,
                )?;
                // Each reference has four independently padded readbacks.
                // All vision outputs remain live until the text bridge copies.
                let bytes = physical(&self.execution.host, &merged)?
                    .checked_mul(1 + Qwen3VLVisionFamily::DEEPSTACK_IDS.len())
                    .context("vision output peak overflows")?;
                vision_outputs = vision_outputs
                    .checked_add(bytes)
                    .context("vision output peak overflows")?;
                // VAE encoders execute and copy one reference at a time.
                encoder_output = encoder_output.max(physical(
                    &self.execution.host,
                    &assets.encoder_config(h, w)?.output_tensor()?,
                )?);
                grids.push(QwenImage21EditImageGrid::new(
                    u64::from(h / patch),
                    u64::from(w / patch),
                    assets.vision.merge_size(),
                )?);
            }
            edit_output = vision_outputs
                .checked_mul(batch)
                .context("vision output peak overflows")?
                .max(encoder_output);
            self.execution
                .sources
                .assets
                .tokenizer
                .prepare_edit_batch_multi(
                    &prompts,
                    self.execution.config.maximum_user_tokens,
                    &grids,
                )?
                .text_config(&self.execution.sources.assets.text)?
        } else {
            let (prompts, _) = self
                .execution
                .sources
                .assets
                .tokenizer
                .prepare_canonical(&prompts, self.execution.config.maximum_user_tokens)?;
            Qwen3VLTextConfig::new(
                self.execution.sources.assets.text.clone(),
                prompts.layout().clone(),
            )?
        };
        // The text graph preserves the embedding geometry through its final
        // residual. This is the full hidden readback, before prefix cropping.
        let text_inputs = text.input_tensor_specs()?;
        let embedding = text_inputs
            .get(&ProgramValueId::new(
                Qwen3VLTextEncoderFamily::EMBEDDINGS_ID,
            )?)
            .context("text embedding geometry is missing")?;
        let hidden = ResolvedTensorSpec::new(
            embedding.dimensions.clone(),
            embedding.element_type,
            embedding.layout.clone(),
        )?;
        let latent =
            QwenImage21LatentShape::new(batch, usize::try_from(height)?, usize::try_from(width)?)?;
        let velocity = latent.tensor(QwenImage21LatentLayout::Packed)?;
        let noise =
            NormalNoiseConfig::new(64, u64::from(height), u64::from(width))?.output_tensor()?;
        let decoder = self.execution.sources.assets.decoder_batch(
            self.execution.config.decoder_batch_policy.batch_size(batch),
            height,
            width,
        )?;
        let decoded = decoder_output_peak(
            physical(&self.execution.host, &decoder.output_tensor()?)?,
            batch,
            self.execution.config.decoder_batch_policy,
        )?;
        Ok(physical(&self.execution.host, &hidden)?
            .max(physical(&self.execution.host, &velocity)?)
            .max(physical(&self.execution.host, &noise)?)
            .max(decoded)
            .max(edit_output))
    }
}
