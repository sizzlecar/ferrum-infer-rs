//! Metadata-only reuse of a verified capacity search, never an admission grant.
use super::*;
use sha2::{Digest, Sha256};

/// The generator's capacity shape for each homogeneous row, independent of
/// batch size and seed. The worker owns one generator, so immutable model,
/// operator, precision and runtime settings cannot cross this cache boundary.
/// Mutable implementations must include their changing configuration in `shape`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageCapacitySearchContext {
    pub shape: [u8; 32],
    pub capacity_bytes: u64,
}

/// One entry bounds metadata even when clients send unbounded shape variants.
/// It retains neither requests, prompts, images nor device allocations.
#[derive(Default)]
pub(super) struct CapacitySearchHint(Option<Boundary>);

struct Boundary {
    shape: [u8; 32],
    adapter: Option<Arc<ImageAdapterBatchIdentity>>,
    rejected: Vec<RejectedBatch>,
}

struct RejectedBatch {
    batch_size: usize,
    required_bytes: u64,
}

impl CapacitySearchHint {
    pub(super) fn upper(
        &self,
        context: Option<ImageCapacitySearchContext>,
        adapter: &Option<Arc<ImageAdapterBatchIdentity>>,
        queued: usize,
    ) -> usize {
        let (Some(context), Some(boundary)) = (context, &self.0) else {
            return queued;
        };
        if boundary.shape != context.shape || boundary.adapter != *adapter {
            return queued;
        }
        // Preserve independent failure bounds when physical capacity changes.
        // A smaller available ceiling must not erase what a previous larger
        // ceiling proved. Reaching a bound permits fresh search, never execution.
        // Prefix exclusion inherits the same-shape monotonicity assumption of
        // the scheduler's binary search; it does not prove arbitrary native
        // workspace choices monotonic or establish a globally optimal batch.
        boundary
            .rejected
            .iter()
            .filter(|bound| bound.required_bytes > context.capacity_bytes)
            .map(|bound| bound.batch_size - 1)
            .fold(queued, usize::min)
    }

    pub(super) fn remember(
        &mut self,
        context: Option<ImageCapacitySearchContext>,
        adapter: &Option<Arc<ImageAdapterBatchIdentity>>,
        count: usize,
        report: super::super::ImageDeviceCapacityReport,
        maximum_batch: usize,
    ) {
        let Some(context) = context else {
            return;
        };
        // Only a typed fresh-plan failure reaches this method. A single-stage
        // lower bound proves failure too. Success, short queues and Host limits
        // are not device bounds. Distinct counts cannot exceed maximum_batch.
        if count <= 1 || count > maximum_batch || report.required_bytes <= report.capacity_bytes {
            return;
        }
        if self
            .0
            .as_ref()
            .is_none_or(|old| old.shape != context.shape || old.adapter != *adapter)
        {
            self.0 = Some(Boundary {
                shape: context.shape,
                adapter: adapter.clone(),
                rejected: Vec::new(),
            });
        }
        let bounds = &mut self.0.as_mut().unwrap().rejected;
        // A smaller rejected batch with an equal or stronger byte lower bound
        // already excludes every candidate that the new observation excludes.
        if bounds
            .iter()
            .any(|old| old.batch_size <= count && old.required_bytes >= report.required_bytes)
        {
            return;
        }
        bounds.retain(|old| {
            !(count <= old.batch_size && report.required_bytes >= old.required_bytes)
        });
        bounds.push(RejectedBatch {
            batch_size: count,
            required_bytes: report.required_bytes,
        });
    }
}

impl<R: DeviceRuntime> QwenImage21Pipeline<R> {
    pub(super) fn service_capacity_search_context(
        &self,
        requests: &[ImageRequestRef<'_>],
    ) -> anyhow::Result<Option<ImageCapacitySearchContext>> {
        // Non-adaptive policies include changing retained owners in the plan's
        // demand. Their existing full search remains authoritative.
        if self.config.residency != super::super::ImageResidencyPolicy::Adaptive {
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
                self.assets
                    .tokenizer
                    .prepare(&prompts, self.config.maximum_user_tokens)?
            }
            ImageRequestRef::Edit(_) => {
                let Some(assets) = &self.assets.editing else {
                    return Ok(None);
                };
                let mut geometry = None;
                for request in requests {
                    let ImageRequestRef::Edit(request) = request else {
                        return Ok(None);
                    };
                    let Some((width, height)) =
                        png_dimensions_hint(request.image.bytes(), ImageInputLimits::default())
                    else {
                        return Ok(None);
                    };
                    let current = QwenImage21ImageProcessor::condition_dimensions(width, height)?;
                    if geometry.is_some_and(|previous| previous != current) {
                        return Ok(None);
                    }
                    geometry = Some(current);
                }
                let (width, height) = geometry.expect("nonempty edit requests");
                let patch = u64::try_from(assets.processor.patch_size())?;
                let prompt = self.assets.tokenizer.prepare_edit_batch(
                    &prompts,
                    self.config.maximum_user_tokens,
                    u64::from(height) / patch,
                    u64::from(width) / patch,
                    assets.vision.merge_size(),
                )?;
                shape.update(b"edit");
                shape.update(width.to_le_bytes());
                shape.update(height.to_le_bytes());
                shape.update(prompt.image_slot_start().to_le_bytes());
                shape.update(prompt.image_slots().to_le_bytes());
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
