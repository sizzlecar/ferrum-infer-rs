//! Image shape projection for the common capacity search.
use super::*;
#[cfg(test)]
pub(crate) use crate::iterative_service::capacity_hint::CapacitySearchHint;
pub use crate::iterative_service::CapacitySearchContext as ImageCapacitySearchContext;
use ferrum_models::qwen_image21::t2i_prompt::QwenImage21EditImageGrid;
use sha2::{Digest, Sha256};
impl<R: DeviceRuntime> QwenImage21Pipeline<R> {
    pub(crate) fn service_capacity_search_context(
        &self,
        requests: &[ImageRequestRef<'_>],
    ) -> anyhow::Result<Option<ImageCapacitySearchContext>> {
        // Non-adaptive policies include changing retained owners in the plan's
        // demand. Their existing full search remains authoritative.
        if self.execution.config.residency != super::super::ImageResidencyPolicy::Adaptive {
            return Ok(None);
        }
        let Some(first) = requests.first() else {
            return Ok(None);
        };
        let generation = first.generation();
        if requests.iter().any(|request| {
            let other = request.generation();
            (other.width, other.height, other.n)
                != (generation.width, generation.height, generation.n)
        }) {
            return Ok(None);
        }
        let mut shape = Sha256::new();
        shape.update(generation.width.to_le_bytes());
        shape.update(generation.height.to_le_bytes());
        shape.update(generation.n.to_le_bytes());
        let prompts = requests
            .iter()
            .map(|request| request.generation().prompt.as_str())
            .collect::<Vec<_>>();
        let prompts = match first {
            ImageRequestRef::Generate(_) => {
                if requests
                    .iter()
                    .any(|request| !matches!(request, ImageRequestRef::Generate(_)))
                {
                    return Ok(None);
                }
                shape.update(b"generate");
                self.execution
                    .sources
                    .assets
                    .tokenizer
                    .prepare(&prompts, self.execution.config.maximum_user_tokens)?
            }
            ImageRequestRef::Edit(_) => {
                let Some(assets) = &self.execution.sources.assets.editing else {
                    return Ok(None);
                };
                let mut geometry: Option<Vec<(u32, u32)>> = None;
                for request in requests {
                    let ImageRequestRef::Edit(request) = request else {
                        return Ok(None);
                    };
                    if !(1..=self.maximum_reference_images()).contains(&request.image_count()) {
                        return Ok(None);
                    }
                    let mut current = Vec::with_capacity(request.image_count());
                    for image in request.images() {
                        let Some((width, height)) =
                            png_dimensions_hint(image.bytes(), ImageInputLimits::default())
                        else {
                            return Ok(None);
                        };
                        assets.processor.prepared_backing_bytes(
                            &self.execution.host,
                            width,
                            height,
                        )?;
                        current.push(QwenImage21ImageProcessor::condition_dimensions(
                            width, height,
                        )?);
                    }
                    if geometry
                        .as_ref()
                        .is_some_and(|previous| previous != &current)
                    {
                        return Ok(None);
                    }
                    geometry = Some(current);
                }
                let geometry = geometry.expect("nonempty edit requests");
                let patch = u64::try_from(assets.processor.patch_size())?;
                let grids = geometry
                    .iter()
                    .map(|&(width, height)| {
                        QwenImage21EditImageGrid::new(
                            u64::from(height) / patch,
                            u64::from(width) / patch,
                            assets.vision.merge_size(),
                        )
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?;
                let prompt = self
                    .execution
                    .sources
                    .assets
                    .tokenizer
                    .prepare_edit_batch_multi(
                        &prompts,
                        self.execution.config.maximum_user_tokens,
                        &grids,
                    )?;
                shape.update(b"edit");
                shape.update(u64::try_from(geometry.len())?.to_le_bytes());
                for ((width, height), slots) in geometry.iter().zip(prompt.image_slot_ranges()) {
                    shape.update(width.to_le_bytes());
                    shape.update(height.to_le_bytes());
                    shape.update(slots.image_slot_start().to_le_bytes());
                    shape.update(slots.image_slots().to_le_bytes());
                }
                prompt.prompts().clone()
            }
        };
        // Equal valid lengths imply the same complete per-row mask and visual
        // span. Different texts with that layout may share a capacity hint;
        // heterogeneous masks use the ordinary search rather than an estimate.
        let length = prompts.valid_input_lengths()[0];
        if prompts
            .valid_input_lengths()
            .iter()
            .any(|&other| other != length)
        {
            return Ok(None);
        }
        shape.update(u64::try_from(prompts.sequence_length())?.to_le_bytes());
        shape.update(u64::try_from(prompts.feature_sequence_length())?.to_le_bytes());
        shape.update(u64::try_from(length)?.to_le_bytes());
        Ok(Some(ImageCapacitySearchContext {
            shape: shape.finalize().into(),
            capacity_bytes: self.adaptive_capacity_ceiling()?,
        }))
    }
}

#[cfg(test)]
mod tests;
