//! Request-owned solver cursors behind the existing synchronous batch API.
//!
//! The caller retains its whole-execution Host grant. Conditioning and the
//! compiled stage still belong to that fixed cohort; this is not a service
//! scheduler or permission to reuse a preparation after changing its members.
use super::*;
use ferrum_diffusion::FlowMatchEulerSchedule;

mod continuous;
#[cfg(test)]
pub(super) use continuous::fixture_request;
pub(crate) use continuous::{ContinuousBatch, ContinuousRequest, PreparedContinuousRequest};
#[cfg(test)]
pub(super) use tests::{pipeline as fixture_pipeline, trajectory as fixture_trajectory};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RequestId {
    job: u64,
    caller_index: usize,
}

struct RequestState {
    id: RequestId,
    sample: ManagedHostTensor,
    solver: ManagedBf16Euler,
}

pub(super) struct FullGenerationDenoising {
    job: u64,
    config: QwenImage21TransformerConfig,
    requests: Vec<RequestState>,
    sample_spec: ResolvedTensorSpec,
    text: ManagedHostTensor,
    frequencies: ManagedHostTensor,
    euler_us: u64,
}

impl FullGenerationDenoising {
    pub(super) fn new(
        host: &HostMemoryAccount,
        job: u64,
        caller_indices: &[usize],
        config: QwenImage21TransformerConfig,
        schedule: &FlowMatchEulerSchedule,
        profile: Bf16EulerProfile,
        sample: ManagedHostTensor,
        text: ManagedHostTensor,
        frequencies: ManagedHostTensor,
    ) -> anyhow::Result<Self> {
        let batch = usize::try_from(config.layout().batch())?;
        ensure!(
            config.layout().reference_image().is_none()
                && config.execution_phase()
                    == ferrum_models::qwen_image21::QwenImage21TransformerPhase::Full,
            "request-owned denoising requires full text-to-image execution"
        );
        ensure!(
            caller_indices.len() == batch
                && caller_indices.iter().all(|&index| index < batch)
                && caller_indices
                    .iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    == batch,
            "denoising identities must preserve the caller permutation"
        );
        let specs = config.input_tensor_specs()?;
        for (tensor, name) in [
            (&sample, QwenImage21TransformerFamily::LATENTS_ID),
            (&text, QwenImage21TransformerFamily::TEXT_ID),
            (&frequencies, QwenImage21TransformerFamily::FREQUENCIES_ID),
        ] {
            let spec = specs
                .get(&id::<ProgramValueId>(name)?)
                .context("missing denoising input")?;
            let expected = ResolvedTensorSpec::new(
                spec.dimensions.clone(),
                spec.element_type,
                spec.layout.clone(),
            )?;
            ensure!(
                tensor.value_id().as_str() == name
                    && tensor.tensor() == &expected
                    && host.owns(tensor.bytes())
                    && tensor.bytes().class() == HostMemoryClass::InputPreprocess,
                "denoising input differs from its admitted cohort"
            );
        }
        let sample_spec = sample.tensor().clone();
        let mut dimensions = sample_spec.dimensions().to_vec();
        ensure!(
            dimensions[0] % batch as u64 == 0,
            "invalid latent row partition"
        );
        dimensions[0] /= batch as u64;
        let item_spec = ResolvedTensorSpec::new(
            dimensions,
            sample_spec.element_type(),
            sample_spec.layout().clone(),
        )?;
        let stride = usize::try_from(item_spec.minimum_storage_bytes()?)?;
        let mut requests = Vec::with_capacity(batch);
        for (&caller_index, source) in caller_indices
            .iter()
            .zip(sample.bytes().bytes().chunks_exact(stride))
        {
            // Separate physical owners let each cursor publish a successor
            // without retaining an obsolete batch-sized latent allocation.
            let mut bytes = host.allocate_zeroed(HostMemoryClass::InputPreprocess, stride)?;
            bytes.as_mut().copy_from_slice(source);
            requests.push(RequestState {
                id: RequestId { job, caller_index },
                sample: ManagedHostTensor::new(
                    sample.value_id().clone(),
                    item_spec.clone(),
                    bytes.freeze(),
                )?,
                solver: ManagedBf16Euler::new(
                    schedule.clone(),
                    host.clone(),
                    item_spec.clone(),
                    profile,
                )?,
            });
        }
        ensure!(requests.len() == batch, "invalid latent byte partition");
        Ok(Self {
            job,
            config,
            requests,
            sample_spec,
            text,
            frequencies,
            euler_us: 0,
        })
    }

    pub(super) fn is_complete(&self) -> bool {
        self.requests
            .iter()
            .all(|request| request.solver.is_complete())
    }

    fn gather(&self, host: &HostMemoryAccount) -> anyhow::Result<ManagedHostTensor> {
        let mut bytes = host.allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            usize::try_from(self.sample_spec.minimum_storage_bytes()?)?,
        )?;
        let stride = self.requests[0].sample.bytes().len();
        for (destination, request) in bytes.as_mut().chunks_exact_mut(stride).zip(&self.requests) {
            ensure!(
                request.id.job == self.job,
                "denoising request belongs to another job"
            );
            destination.copy_from_slice(request.sample.bytes().bytes());
        }
        Ok(ManagedHostTensor::new(
            id(QwenImage21TransformerFamily::LATENTS_ID)?,
            self.sample_spec.clone(),
            bytes.freeze(),
        )?)
    }

    /// Execute one real model evaluation. The returned active stage is the
    /// original execution owner; every submitted failure retains it in the
    /// existing typed failure. Request cursors publish only after all rows
    /// have admitted and calculated their successor samples successfully.
    pub(super) fn step<R: DeviceRuntime>(
        &mut self,
        pipeline: &mut QwenImage21Pipeline<R>,
        active: ActiveImageStage<R>,
        steps: &mut Vec<ImageDenoiseStepReceipt>,
    ) -> Result<ActiveImageStage<R>, ImageGenerationFailure<R>> {
        let inputs = (|| -> anyhow::Result<_> {
            let times = self
                .requests
                .iter()
                .map(|request| {
                    request
                        .solver
                        .current_step()
                        .map(|step| step.timestep().as_f32())
                        .context("completed request cannot consume another denoising evaluation")
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            Ok(vec![
                self.gather(pipeline.execution_host())?,
                self.text.clone(),
                self.config
                    .managed_time_features_per_sample(pipeline.execution_host(), &times)?,
                self.frequencies.clone(),
            ])
        })();
        let inputs = match inputs {
            Ok(inputs) => inputs,
            Err(error) => return Err(active.processing(error)),
        };
        // Keep the immutable submitted input snapshot until every successor
        // has prepared. The Host grant explicitly includes this gathered batch
        // alongside all old and new independently owned request samples.
        let (active, outputs) = pipeline.execute_stage(active, self.job, inputs.clone())?;
        let updated = (|| -> anyhow::Result<_> {
            let velocity = only_output(outputs, QwenImage21TransformerFamily::VELOCITY_ID)?;
            ensure!(
                velocity.tensor() == &self.sample_spec,
                "velocity differs from the admitted latent tensor"
            );
            let stride = self.requests[0].sample.bytes().len();
            let started = std::time::Instant::now();
            let pending = self
                .requests
                .iter_mut()
                .enumerate()
                .map(|(index, request)| {
                    let row = velocity
                        .bytes()
                        .slice(index * stride..(index + 1) * stride)?;
                    Ok(request.solver.prepare_advance(
                        &request.sample,
                        request.sample.tensor(),
                        &row,
                    )?)
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            let updated = pending
                .into_iter()
                .map(|pending| pending.commit())
                .collect::<Vec<_>>();
            self.euler_us = self.euler_us.saturating_add(trace_elapsed_us(started));
            Ok(updated)
        })();
        let updated = match updated {
            Ok(updated) => updated,
            Err(error) => return Err(active.processing(error)),
        };
        // Fixed-cohort callers use one common schedule. Preserve their public
        // batch receipt while retaining independent cursors internally.
        let step = updated[0].0;
        for (request, (_, sample)) in self.requests.iter_mut().zip(updated) {
            request.sample = sample;
        }
        steps.push(ImageDenoiseStepReceipt {
            index: step.index(),
            sigma: step.sigma(),
            next_sigma: step.next_sigma(),
            scheduler_timestep: step.timestep().as_f32(),
        });
        Ok(active)
    }

    pub(super) fn finish(self, host: &HostMemoryAccount) -> anyhow::Result<ManagedHostTensor> {
        ensure!(
            self.is_complete(),
            "cannot decode incomplete denoising requests"
        );
        tracing::debug!(target: "ferrum_engine::image_generation::timing", job = self.job,
            euler_us = self.euler_us, "image denoising host solver completed");
        self.gather(host)
    }
}

#[cfg(test)]
mod tests;
