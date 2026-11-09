//! Synchronous generation consumes the same request lanes and packed step
//! owner as the service worker; it has no separate per-request DiT loop.
use super::*;
use crate::image_generation::service::{ImageBatchError, ImageRequestRef};
use crate::image_generation::{
    ImageEditRequest, ImageGenerationError, ImageGenerationOutput, ImageGenerationRequest,
};

impl<R: DeviceRuntime> PH3ImagePipeline<R> {
    pub(in crate::image_generation) fn generate_continuous_batch(
        &mut self,
        requests: Vec<ImageGenerationRequest>,
    ) -> std::result::Result<Vec<ImageGenerationOutput>, ImageGenerationError<R>> {
        self.execute_continuous_requests(
            &requests
                .iter()
                .map(ImageRequestRef::Generate)
                .collect::<Vec<_>>(),
        )
    }
    pub(in crate::image_generation) fn edit_continuous_batch(
        &mut self,
        requests: Vec<ImageEditRequest>,
    ) -> std::result::Result<Vec<ImageGenerationOutput>, ImageGenerationError<R>> {
        self.execute_continuous_requests(
            &requests
                .iter()
                .map(ImageRequestRef::Edit)
                .collect::<Vec<_>>(),
        )
    }
    fn execute_continuous_requests(
        &mut self,
        requests: &[ImageRequestRef<'_>],
    ) -> std::result::Result<Vec<ImageGenerationOutput>, ImageGenerationError<R>> {
        let preflight = |source| ImageGenerationError {
            failure: ImageGenerationFailure::Preflight(source),
            completed_stages: Vec::new(),
            completed_steps: Vec::new(),
        };
        let validate = || -> Result<_> {
            ensure!(
                self.dit_input_diagnostics.is_none() || requests.len() == 1,
                "DiT input capture requires B1 before request preparation"
            );
            ensure!(
                !self.is_stopped()
                    && self.execution.execution_host.is_none()
                    && !requests.is_empty()
                    && requests.len() <= self.execution.config.maximum_batch_size,
                "H3 generation requires a bounded nonempty batch on a live idle pipeline"
            );
            Ok(())
        };
        validate().map_err(preflight)?;
        let preparations = requests
            .iter()
            .map(|request| self.prepare_initialization(*request))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(initialization_failure)?;
        let preparations = self
            .prepare_run_initializations(preparations, requests)
            .map_err(initialization_failure)?;
        let mut lanes = Vec::<ContinuousRequest<R>>::with_capacity(requests.len());
        let mut batch = None::<ContinuousBatch<R>>;
        let result = (|| {
            for (prepared, request) in preparations.into_iter().zip(requests) {
                if let Some(previous) = batch.as_mut() {
                    previous
                        .pause(self, &mut lanes.iter_mut().collect::<Vec<_>>())
                        .map_err(operation_failure)?;
                }
                let initialized = match prepared {
                    expansion::PreparedRunInitialization::Individual(mut prepared) => {
                        let admitted = self
                            .admit_initialization_at_boundary(
                                &mut prepared,
                                0,
                                batch.as_mut(),
                                &mut lanes.iter_mut().collect::<Vec<_>>(),
                            )
                            .map_err(initialization_failure)?;
                        if !admitted {
                            return Err(preflight(anyhow::anyhow!(
                        "H3 initial request cannot obtain persistent and first-forward work credit"
                    )));
                        }
                        self.start_initialization_blocking(prepared, *request)
                    }
                    expansion::PreparedRunInitialization::Expanded(initialization) => {
                        self.continue_initialization_blocking(initialization, *request)
                    }
                }
                .and_then(|request| request.into_inner::<ContinuousRequest<R>>());
                if matches!(&initialized, Err(ImageBatchError::Fatal(_))) {
                    return Err(initialization_failure(initialized.err().unwrap()));
                }
                // A normal request rejection/cancellation never leaves old
                // peers attached to a retired installation. Unknown native
                // cleanup above remains fatal and keeps every original owner.
                if let Some(previous) = batch.as_mut().filter(|batch| batch.suspended) {
                    self.restore_continuous_batch(
                        previous,
                        &mut lanes.iter_mut().collect::<Vec<_>>(),
                    )
                    .map_err(operation_failure)?;
                }
                lanes.push(initialized.map_err(initialization_failure)?);
                let mut members = lanes.iter_mut().collect::<Vec<_>>();
                let next = if let Some(previous) = batch.as_mut() {
                    self.try_replace_continuous_batch(previous, &mut members)
                } else {
                    self.try_prepare_continuous_batch(&mut members)
                }
                .map_err(operation_failure)?;
                batch = Some(next.ok_or_else(|| {
                    preflight(anyhow::anyhow!(
                        "H3 initial cohort cannot obtain actual native and step work capacity"
                    ))
                })?);
            }
            while lanes.iter().any(|lane| !lane.is_complete()) {
                batch
                    .as_mut()
                    .expect("nonempty initialized H3 batch")
                    .advance(self, &mut lanes.iter_mut().collect::<Vec<_>>())
                    .map_err(operation_failure)?;
            }
            batch
                .as_mut()
                .expect("initialized H3 batch")
                .pause(self, &mut lanes.iter_mut().collect::<Vec<_>>())
                .map_err(operation_failure)?;
            // All lanes have left denoising at the original pause fence. A
            // later decoder failure must not erase another lane's real exit.
            for lane in &lanes {
                self.trace_request_exit(
                    lane,
                    crate::image_generation::service::ImageDenoisingExitReason::Completed,
                );
            }
            let mut outputs = Vec::with_capacity(lanes.len());
            for lane in &mut lanes {
                outputs.push(
                    self.finish_continuous_request_borrowed(
                        lane,
                        batch.as_mut().expect("paused H3 decoder work owner"),
                    )
                    .map_err(operation_failure)?,
                );
            }
            Ok(outputs)
        })();
        match result {
            Ok(outputs) => Ok(outputs),
            Err(source) => {
                if lanes.is_empty()
                    && batch.is_none()
                    && matches!(source.failure, ImageGenerationFailure::Preflight(_))
                {
                    // Invalid ingress before the first initialized lane has
                    // no native recovery owner and does not poison the owner.
                    return Err(source);
                }
                self.execution.stopped = true;
                Err(retain_failure(source, lanes, batch))
            }
        }
    }
}

fn initialization_failure<R: DeviceRuntime>(error: ImageBatchError) -> ImageGenerationError<R> {
    let failure = match error {
        ImageBatchError::Fatal(source) => ImageGenerationFailure::InitializationPhase { source },
        ImageBatchError::Capacity { proof, diagnostic } => {
            ImageGenerationFailure::Preflight(anyhow::Error::new(proof).context(diagnostic))
        }
        other => ImageGenerationFailure::Preflight(anyhow::anyhow!(other.to_string())),
    };
    ImageGenerationError {
        failure,
        completed_stages: vec![],
        completed_steps: vec![],
    }
}

fn operation_failure<R: DeviceRuntime>(
    failure: ImageGenerationFailure<R>,
) -> ImageGenerationError<R> {
    ImageGenerationError {
        failure,
        completed_stages: Vec::new(),
        completed_steps: Vec::new(),
    }
}

/// All existing peers and grants survive a synchronous failure, including a
/// clean-but-rejected late cohort replacement. No replacement owner is forged.
fn retain_failure<R: DeviceRuntime>(
    mut source: ImageGenerationError<R>,
    lanes: Vec<ContinuousRequest<R>>,
    batch: Option<ContinuousBatch<R>>,
) -> ImageGenerationError<R> {
    let mut peers = Vec::new();
    let mut weights = Vec::new();
    let mut scopes = Vec::new();
    let mut backing = Vec::new();
    if let Some(mut batch) = batch {
        if let Some(active) = batch.active.take() {
            peers.push(active);
        }
        if let Some(work) = batch.work.take() {
            scopes.push(work);
        }
        source.completed_stages.extend(batch.stages);
        backing.extend(batch.phase_backing.into_values());
    }
    for mut lane in lanes {
        if let Some(pin) = lane.weights.take() {
            weights.push(pin);
        }
        scopes.push(lane.persistent);
        if let Some(work) = lane.minimum_work.take() {
            scopes.push(work);
        }
        backing.extend(lane.backing);
        source.completed_stages.extend(lane.stages);
        source.completed_steps.extend(lane.steps);
    }
    ImageGenerationError {
        failure: ImageGenerationFailure::Pipeline {
            source: Box::new(source.failure),
            stages: peers,
            weights,
            host_scopes: scopes,
            host_backing: backing,
        },
        completed_stages: source.completed_stages,
        completed_steps: source.completed_steps,
    }
}
