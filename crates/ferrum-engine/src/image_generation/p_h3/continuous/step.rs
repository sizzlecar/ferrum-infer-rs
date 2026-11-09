//! The shared production step consumes borrowed request lanes. This boundary
//! also permits a real tiny DiT fixture without inventing TE/VAE asset stubs.
use super::*;
use crate::image_generation::{ImagePipelineExecution, ImagePipelineSourceAdapter};
use ferrum_models::p_h3_image::{
    PH3ImageDenoiserPhase, PH3ImageDenoiserReference, PH3ImageSamplingStep,
};

pub(super) struct StepLane<'a> {
    pub identity: &'a RequestIdentity,
    pub image: &'a mut ManagedHostTensor,
    pub audio: &'a mut ManagedHostTensor,
    pub sampling: &'a mut PH3ImageManagedSampling,
    pub cache: &'a PH3ImageRefinedTextCache,
    pub references: Vec<PH3ImageDenoiserReference<'a>>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute<R: DeviceRuntime, S: ImagePipelineSourceAdapter>(
    execution: &mut ImagePipelineExecution<R, S>,
    prepared: &CompiledSteady<R>,
    mut active: ActiveImageStage<R>,
    job: u64,
    host: &HostMemoryAccount,
    weights: &SharedComponentWeights<R>,
    times: &[f32],
    positions: &[[f64; 3]],
    lanes: &mut [StepLane<'_>],
    diagnostics: Option<(
        &mut super::super::dit_input_diagnostics::DitInputDiagnostics,
        super::super::dit_input_diagnostics::DitInputIdentity,
    )>,
) -> std::result::Result<(ActiveImageStage<R>, Vec<PH3ImageSamplingStep>), ImageGenerationFailure<R>>
{
    let inputs = (|| -> Result<_> {
        ensure!(
            !lanes.is_empty()
                && prepared.recipe.phase() == PH3ImageDenoiserPhase::DenoiseRefinedText
                && prepared.recipe.request_ids()
                    == lanes
                        .iter()
                        .map(|lane| lane.identity.clone())
                        .collect::<Vec<_>>()
                && lanes.iter().all(|lane| !lane.sampling.is_complete()),
            "H3 step requires its exact ordered, incomplete request lanes"
        );
        prepared
            .recipe
            .validate_attached_session(&active.session, weights)?;
        let samples = lanes
            .iter()
            .map(|lane| PH3ImageDenoiserSample {
                image: &*lane.image,
                audio: &*lane.audio,
            })
            .collect::<Vec<_>>();
        let caches = lanes
            .iter()
            .map(|lane| lane.cache.clone())
            .collect::<Vec<_>>();
        prepared.recipe.prepare_inputs_with_references(
            host,
            &samples,
            &lanes
                .iter()
                .map(|lane| lane.references.clone())
                .collect::<Vec<_>>(),
            &caches,
            weights.evidence(),
            times,
            positions,
        )
    })();
    let inputs = match inputs {
        Ok(inputs) => inputs,
        Err(source) => return Err(active.processing(source)),
    };
    if let Some((diagnostics, identity)) = diagnostics {
        let observed = (|| -> Result<()> {
            ensure!(lanes.len() == 1, "DiT input capture requires B1");
            let lane = &lanes[0];
            let runs = prepared
                .recipe
                .request_row_runs()
                .next()
                .context("missing B1 routing")?;
            diagnostics.deliver(
                host,
                super::super::PH3ImageDitInputObservation {
                    request: lane.identity,
                    job: identity.job,
                    cohort_job: job,
                    seed: identity.seed,
                    width: identity.width,
                    height: identity.height,
                    step: lane
                        .sampling
                        .current_step()
                        .context("completed DiT capture lane")?,
                    family: prepared.recipe.family(),
                    row_runs: runs,
                    times,
                    inputs: &inputs,
                    image_state: lane.image,
                    audio_state: lane.audio,
                },
            )
        })();
        if let Err(source) = observed {
            // The original active owner follows the existing processing-error
            // custody. No actor execution or sampler commit has happened.
            return Err(active.processing(source));
        }
    }
    if let Err(source) = active.session.set_output_host_account(host.clone()) {
        return Err(active.processing(source.into()));
    }
    if execution.serial_actor.is_none() {
        return Err(active.processing(anyhow::anyhow!("H3 serial actor is closed")));
    }
    // This is the original actor call boundary, not a GPU-submit timestamp.
    // Starts whose execution/atomic successor preparation fails have no commit.
    let start_sequence = super::trace::started(execution, prepared, job, lanes);
    let Some(actor) = execution.serial_actor.as_mut() else {
        unreachable!("actor checked above");
    };
    let (active, outputs) = active.execute_on_actor(actor, job, inputs)?;
    let successors = (|| -> Result<_> {
        ensure!(
            outputs.len() == 2,
            "H3 step requires image and audio velocities"
        );
        let ports = host_budget::ports(&prepared.stage)?;
        let velocity = prepared.recipe.velocity_layout()?;
        let output = |name: &str| -> Result<_> {
            let port = ports
                .outputs()
                .iter()
                .find(|p| p.value_id().as_str() == name)
                .context("missing declared H3 step velocity")?;
            let output = outputs
                .iter()
                .find(|p| p.value_id().as_str() == name)
                .context("missing completed H3 step velocity")?;
            Ok((port, output))
        };
        let (ip, iv) = output(PH3ImageFinalLayerFamily::IMAGE_VELOCITY_ID)?;
        let (ap, av) = output(PH3ImageFinalLayerFamily::AUDIO_VELOCITY_ID)?;
        let pending = lanes
            .iter_mut()
            .map(|lane| {
                lane.sampling.prepare_advance(
                    lane.image,
                    velocity.image_output(lane.identity, ip, iv)?,
                    lane.audio,
                    velocity.audio_output(lane.identity, ap, av)?,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        // Every allocation, mapped read and numerical check has succeeded for
        // all streams before the first exclusively borrowed cursor commits.
        Ok(pending
            .into_iter()
            .map(|advance| advance.commit())
            .collect::<Vec<_>>())
    })();
    let successors = match successors {
        Ok(successors) => successors,
        Err(source) => return Err(active.processing(source)),
    };
    let mut steps = Vec::with_capacity(lanes.len());
    for (lane, (step, image, audio)) in lanes.iter_mut().zip(successors) {
        *lane.image = image;
        *lane.audio = audio;
        steps.push(step);
    }
    // The original actor fence and every successor commit have succeeded.
    // Diagnostic failure cannot turn this completed forward into a retry.
    super::trace::committed(
        execution,
        prepared,
        job,
        times,
        lanes,
        &steps,
        start_sequence,
    );
    drop(outputs);
    Ok((active, steps))
}
