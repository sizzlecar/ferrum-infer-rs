//! Two fixed tensor plans, one immutable weight installation, one request cache.
//! The prefill outputs are copied on-device into admitted Request state, then
//! its Step arena retires before the target-only plan starts executing.
use super::*;
use ferrum_execution::{TensorRequestSession, TensorRequestStateMapping};
use std::time::Instant;

pub(super) struct PrefixDenoise<R: DeviceRuntime> {
    pub prefill: PreparedStage<R>,
    pub decode: prepare::PreparedPrefixDecode<R>,
    pub config: QwenImage21TransformerConfig,
    pub sample: ManagedHostTensor,
    pub text: ManagedHostTensor,
    pub reference: ManagedHostTensor,
    pub frequencies: ManagedHostTensor,
}

struct Owners<R: DeviceRuntime> {
    prefill: Option<ActiveImageStage<R>>,
    decode: Option<ActiveImageStage<R>>,
    request: Option<TensorRequestSession<R>>,
    weights: Option<SharedComponentWeights<R>>,
}
impl<R: DeviceRuntime> Owners<R> {
    fn fail(self, source: ImageGenerationFailure<R>) -> ImageGenerationFailure<R> {
        ImageGenerationFailure::Prefix {
            source: Box::new(source),
            request: self.request,
            stages: self.prefill.into_iter().chain(self.decode).collect(),
            weights: self.weights,
        }
    }
}

impl<R: DeviceRuntime> QwenImage21Pipeline<R> {
    pub(super) fn run_prefix_denoising(
        &mut self,
        work: PrefixDenoise<R>,
        solver: &mut ManagedBf16Euler,
        job: u64,
        stages: &mut Vec<ImageStageReceipt>,
        steps: &mut Vec<ImageDenoiseStepReceipt>,
    ) -> Result<ManagedHostTensor, ImageGenerationFailure<R>> {
        let process = |source| ImageGenerationFailure::Processing {
            source,
            active: None,
        };
        let PrefixDenoise {
            prefill,
            decode,
            config,
            mut sample,
            text,
            reference,
            frequencies,
        } = work;
        let decode_frequencies = decode
            .config
            .managed_rotary_frequencies(&self.host)
            .map_err(process)?;
        let mappings = config
            .prefix_cache_ports()
            .map_err(|e| process(e.into()))?
            .into_iter()
            .map(|port| TensorRequestStateMapping {
                source_output: port.value_id().clone(),
                destination_state: port.value_id().clone(),
            })
            .collect::<Vec<_>>();
        let first = solver
            .current_step()
            .context("prefix denoising requires a first timestep")
            .map_err(process)?;
        let initial_time = decode
            .config
            .managed_time_features(&self.host, first.timestep().as_f32())
            .map_err(process)?;
        let run = id(format!("run.image.{job}.denoise.prefix")).map_err(process)?;
        let request = id(format!("request.image.{job}.denoise.prefix")).map_err(process)?;
        let (prefill, weights, temporary_weights) =
            self.start_prefix_prefill(prefill, job, stages)?;
        let mut owners = Owners {
            prefill: Some(prefill),
            decode: None,
            request: None,
            weights: Some(weights),
        };
        let result = (|| -> Result<ManagedHostTensor, ImageGenerationFailure<R>> {
            owners.decode = Some(decode.stage.start_attached(
                self,
                job,
                owners.weights.as_ref().unwrap(),
            )?);
            let request_session = owners
                .decode
                .as_ref()
                .unwrap()
                .session
                .begin_request(
                    run,
                    request,
                    vec![sample.clone(), initial_time, decode_frequencies.clone()],
                )
                .map_err(|source| ImageGenerationFailure::RequestExecution { source })?;
            owners.request = Some(request_session);
            let initial_inputs = vec![
                sample.clone(),
                text,
                config
                    .managed_time_features(&self.host, first.timestep().as_f32())
                    .map_err(process)?,
                frequencies,
                reference,
            ];
            let prefill = owners.prefill.as_mut().unwrap();
            let started = Instant::now();
            let output = prefill.session.execute_initializing_request(
                id(format!("run.image.{job}.denoise.prefill")).map_err(process)?,
                id(format!("request.image.{job}.denoise.prefill")).map_err(process)?,
                initial_inputs,
                owners.request.as_mut().unwrap(),
                &mappings,
            );
            prefill
                .receipt
                .execution_wall_durations_us
                .push(elapsed_us(started));
            let output =
                output.map_err(|source| ImageGenerationFailure::RequestExecution { source })?;
            prefill.receipt.completed_executions += 1;
            advance(solver, &mut sample, output, steps).map_err(process)?;
            // Outputs no longer borrow the prefill arena after the D2D fence.
            // Explicitly retire it before admitting any decode Step backing.
            stages.push(owners.prefill.take().unwrap().close()?);
            while let Some(step) = solver.current_step() {
                let time = decode
                    .config
                    .managed_time_features(&self.host, step.timestep().as_f32())
                    .map_err(process)?;
                let started = Instant::now();
                let output = owners.request.as_mut().unwrap().execute(vec![
                    sample.clone(),
                    time,
                    decode_frequencies.clone(),
                ]);
                let active = owners.decode.as_mut().unwrap();
                active
                    .receipt
                    .execution_wall_durations_us
                    .push(elapsed_us(started));
                let output =
                    output.map_err(|source| ImageGenerationFailure::RequestExecution { source })?;
                active.receipt.completed_executions += 1;
                advance(solver, &mut sample, output, steps).map_err(process)?;
            }
            // The cache cannot overlap the VAE decoder's dynamic peak.
            owners
                .request
                .take()
                .unwrap()
                .close()
                .map_err(|source| ImageGenerationFailure::RequestClose { source })?;
            let origin = owners.decode.as_ref().unwrap().receipt.clone();
            self.finish_stage(owners.decode.take().unwrap(), stages)?;
            let weights = owners.weights.take().unwrap();
            if temporary_weights {
                let started = Instant::now();
                let receipt = match weights.close() {
                    Ok(SharedComponentWeightsCloseOutcome::Closed(receipt)) => receipt,
                    Ok(outcome) => {
                        return Err(ImageGenerationFailure::SharedWeightsReferenced {
                            stage: ImageStage::Denoiser,
                            outcome,
                        })
                    }
                    Err(source) => {
                        return Err(ImageGenerationFailure::SharedWeightsClose {
                            stage: ImageStage::Denoiser,
                            source,
                        })
                    }
                };
                let mut origin = origin;
                origin.completed_executions = 0;
                origin.execution_wall_durations_us.clear();
                origin.initialization_wall_duration_us = 0;
                origin.plan.compile_wall_duration_us = 0;
                stages.push(ImageStageReceipt {
                    execution: origin,
                    close: None,
                    retirement: ImageStageRetirement::SharedWeightsClosed { receipt },
                    close_wall_duration_us: elapsed_us(started),
                });
            }
            Ok(sample)
        })();
        result.map_err(|source| owners.fail(source))
    }
}

fn elapsed_us(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
}

fn advance(
    solver: &mut ManagedBf16Euler,
    sample: &mut ManagedHostTensor,
    outputs: Vec<TensorExecutionOutput>,
    steps: &mut Vec<ImageDenoiseStepReceipt>,
) -> anyhow::Result<()> {
    let step = solver
        .current_step()
        .context("denoising schedule exhausted")?;
    let velocity = only_output(outputs, QwenImage21TransformerFamily::VELOCITY_ID)?;
    let (_, next) = solver.advance(sample, velocity.tensor(), velocity.bytes())?;
    *sample = next;
    steps.push(ImageDenoiseStepReceipt {
        index: step.index(),
        sigma: step.sigma(),
        next_sigma: step.next_sigma(),
        scheduler_timestep: step.timestep().as_f32(),
    });
    Ok(())
}
