//! Decoder work reuses the fenced cohort grant. The final pixel allocation is
//! charged once and may outlive this work scope through its ordinary owner.
use super::*;
use crate::image_generation::{ImageGenerationOutput, ImageGenerationReceipt};

impl<R: DeviceRuntime> PH3ImagePipeline<R> {
    pub(in crate::image_generation) fn finish_continuous_request_borrowed(
        &mut self,
        request: &mut ContinuousRequest<R>,
        batch: &mut ContinuousBatch<R>,
    ) -> std::result::Result<ImageGenerationOutput, ImageGenerationFailure<R>> {
        let prepare = || -> Result<_> {
            ensure!(
                !self.is_stopped()
                    && self.execution.execution_host.is_none()
                    && request.is_complete()
                    && !request.decoded
                    && batch.active.is_none()
                    && Arc::ptr_eq(&request.pipeline, &batch.pipeline)
                    && Arc::ptr_eq(
                        &batch.pipeline,
                        &self.execution.service_preparation_identity,
                    ),
                "H3 decoding requires a completed request and a fenced same-pipeline cohort"
            );
            // This lane's actual B1 recipe remains independent of a newly
            // admitted survivor membership. It cannot become a new step row.
            let report = request
                .decoder
                .recipe
                .host_requirements(&self.execution.host)?;
            let stages = request.decoder.stages.iter().collect::<Vec<_>>();
            let (weights, backing) = self.source_backing_budget(&stages)?;
            let staging = self.static_staging_budget(&stages)?;
            let budget = ExecutionHostBudget {
                input: report.input_peak_bytes,
                output: report.output_peak_bytes,
                staging,
                weights,
            };
            self.execution
                .check_scope_capacity_with_retained_diagnostics(&budget.claims())?;
            Ok((budget, backing))
        };
        let (budget, backing) = prepare().map_err(ImageGenerationFailure::Preflight)?;
        let old = batch
            .work
            .take()
            .context("H3 decoder work authority was transferred")
            .map_err(process)?;
        let scope = match self.execution.host.exchange_scope(old, &budget.claims()) {
            Ok(scope) => scope,
            Err(failure) => {
                let (source, old) = failure.into_parts();
                batch.work = Some(old);
                return Err(ImageGenerationFailure::Preflight(source.into()));
            }
        };
        // Completed requests no longer require DiT attachments. All still-live
        // lanes retain their own native pins; normal adaptive admission decides
        // which unpinned idle installation can actually retire for the VAE.
        request.weights.take();
        self.execution.execution_host = Some(scope.account());
        let output = self.execute_decoder(
            &request.decoder,
            request.image.clone(),
            request.job,
            &request.geometry.identity,
            &mut request.stages,
        );
        self.execution.execution_host = None;
        batch.work = Some(scope);
        let rgba = match output {
            Ok(pixels) => pixels,
            Err(source) => {
                return Err(ImageGenerationFailure::Pipeline {
                    source: Box::new(source),
                    stages: Vec::new(),
                    weights: Vec::new(),
                    host_scopes: Vec::new(),
                    host_backing: backing,
                });
            }
        };
        drop(backing);
        request.decoded = true;
        // Diagnostic failure preserves the already-completed decoded state.
        // Processing follows the caller's existing failure-custody path.
        super::super::final_latent_diagnostics::observe_final_latent(
            &mut self.final_latent_diagnostics,
            &self.execution.host,
            &request.geometry.identity,
            request.job,
            request.width,
            request.height,
            request.seed,
            request.steps.len(),
            &request.image,
        )
        .map_err(process)?;
        Ok(ImageGenerationOutput {
            rgba,
            receipt: ImageGenerationReceipt {
                job: request.job,
                batch_size: request.last_batch.0,
                batch_index: request.last_batch.1,
                decoder_batch_policy: self.execution.config.decoder_batch_policy,
                residency: self.execution.config.residency,
                width: request.width,
                height: request.height,
                seed: request.seed,
                stages: request.stages.clone(),
                steps: request.steps.clone(),
                runtime_fingerprint: self
                    .execution
                    .runtime
                    .descriptor()
                    .runtime_implementation_fingerprint
                    .clone(),
                native_memory: self.execution.runtime.native_memory_snapshot(),
                assets: self.execution.sources.assets.receipt().clone(),
                euler_profile: PH3ImageManagedSampling::NUMERICAL_PROFILE_ID,
                noise_layout_profile: Some(self.options.initial_noise.identity()),
            },
        })
    }
}
