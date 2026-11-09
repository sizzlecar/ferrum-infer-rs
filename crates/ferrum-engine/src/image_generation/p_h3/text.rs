//! Stage-by-stage H3 conditioning over the shared execution owner. Core Host
//! arithmetic remains in the model adapters; each real component submission
//! goes through ActiveImageStage, retaining its own timing/profile receipt.
use super::{conditioning::PreparedPrompts, stages::CompiledConditioning, PH3ImagePipeline};
use crate::image_generation::{
    ActiveImageStage, ImageGenerationFailure, ImagePipelineExecution, ImagePipelineSourceAdapter,
    ImageStage, ImageStageReceipt,
};
use anyhow::{ensure, Context, Result};
use ferrum_execution::TensorExecutionOutput;
use ferrum_models::{
    p_h3_image::{
        PH3ImageConditioningPrecision, PH3ImageF32HiddenState, PH3ImageTextConditioningConfig,
        PH3ImageTextConnectorAffineArithmetic, PH3ImageTextConnectorCheckpoint,
        PH3ImageTextConnectorCheckpointLayout, PH3ImageTextConnectorConfig,
        PH3ImageTextConnectorManagedAdapter,
    },
    qwen3_vl_text::Qwen3VLTextComfyFamily,
    qwen_image21::image_input::ManagedRgba8Image,
};
use ferrum_runtime::*;
use std::sync::Arc;

impl<R: DeviceRuntime> PH3ImagePipeline<R> {
    pub(super) fn execute_conditioning(
        &mut self,
        prepared: &CompiledConditioning<R>,
        job: u64,
        receipts: &mut Vec<ImageStageReceipt>,
    ) -> std::result::Result<ManagedHostTensor, ImageGenerationFailure<R>> {
        self.execute_conditioning_with_image(prepared, None, job, receipts)
    }

    /// The caller supplies the admitted, explicitly preprocessed image only
    /// at execution. Pure-text preparation never needs an image payload.
    pub(super) fn execute_conditioning_with_image(
        &mut self,
        prepared: &CompiledConditioning<R>,
        image: Option<&ManagedRgba8Image>,
        job: u64,
        receipts: &mut Vec<ImageStageReceipt>,
    ) -> std::result::Result<ManagedHostTensor, ImageGenerationFailure<R>> {
        self.execute_conditioning_with_image_reuse(prepared, image, job, receipts, None)
    }

    pub(super) fn execute_conditioning_with_image_reuse(
        &mut self,
        prepared: &CompiledConditioning<R>,
        image: Option<&ManagedRgba8Image>,
        job: u64,
        receipts: &mut Vec<ImageStageReceipt>,
        reuse: Option<&super::vision::reuse::MainVisionReuseBinding>,
    ) -> std::result::Result<ManagedHostTensor, ImageGenerationFailure<R>> {
        self.execute_conditioning_with_images_reuse(
            prepared,
            image.map(std::slice::from_ref),
            job,
            receipts,
            reuse,
        )
    }

    pub(super) fn execute_conditioning_with_images_reuse(
        &mut self,
        prepared: &CompiledConditioning<R>,
        image: Option<&[ManagedRgba8Image]>,
        job: u64,
        receipts: &mut Vec<ImageStageReceipt>,
        reuse: Option<&super::vision::reuse::MainVisionReuseBinding>,
    ) -> std::result::Result<ManagedHostTensor, ImageGenerationFailure<R>> {
        let span = self.trace_phase_started(
            None,
            Some(&prepared.recipe.geometry.identity),
            crate::image_generation::continuous_trace::ImageTracePhase::Conditioning,
        );
        let result =
            self.execute_conditioning_with_image_inner(prepared, image, job, receipts, reuse);
        self.trace_phase_finished(
            span,
            if result.is_ok() {
                crate::image_generation::continuous_trace::ImageTracePhaseOutcome::Completed
            } else {
                crate::image_generation::continuous_trace::ImageTracePhaseOutcome::Failed
            },
        );
        result
    }

    fn execute_conditioning_with_image_inner(
        &mut self,
        prepared: &CompiledConditioning<R>,
        image: Option<&[ManagedRgba8Image]>,
        job: u64,
        receipts: &mut Vec<ImageStageReceipt>,
        reuse: Option<&super::vision::reuse::MainVisionReuseBinding>,
    ) -> std::result::Result<ManagedHostTensor, ImageGenerationFailure<R>> {
        if self.execution.is_stopped() {
            return Err(ImageGenerationFailure::Unavailable);
        }
        let assets = Arc::clone(&self.execution.sources.assets);
        let recipe = &prepared.recipe;
        let host = self.execution.execution_host().clone();
        let prepare = || -> Result<_> {
            ensure!(
                prepared.encoder.declaration().stage == ImageStage::TextEncoder
                    && prepared.connector.declaration().stage == ImageStage::TextConnector
                    && prepared.encoder.family().fingerprint()? == recipe.encoder.fingerprint()?
                    && prepared.connector.family().fingerprint()?
                        == recipe.connector.fingerprint()?,
                "conditioning stages differ from their validated component recipes"
            );
            validate_connector_domain(&recipe.config.connector_config()?, &assets.connector)?;
            // This keeps the existing checkpoint validation contract. It
            // borrows original mmap bytes and never expands/copies weights.
            let checkpoint = PH3ImageTextConnectorCheckpoint::new_with_layout(
                &assets.connector,
                assets.sources.text_encoder.archive(),
                assets.limits.connector,
                PH3ImageTextConnectorCheckpointLayout::ComfyMergedTextEncoder,
            )?;
            let adapter = PH3ImageTextConnectorManagedAdapter::new(
                &assets.connector,
                recipe.config.encoder().text().layout().batch(),
                recipe.config.encoder().text().layout().sequence_length() as usize,
            )?
            .with_output_affine(PH3ImageTextConnectorAffineArithmetic::FusedAddcmulF32);
            ensure!(
                recipe.prompts.visual().is_some() == prepared.visual.is_some()
                    && prepared.visual.is_some() == image.is_some(),
                "conditioning presentation, visual stage and admitted image presence differ"
            );
            Ok((checkpoint, adapter))
        };
        let (checkpoint, adapter) = prepare()
            .map_err(ImageGenerationFailure::Preflight)
            .map_err(pipeline_failure)?;
        let inputs = match (&recipe.prompts, &prepared.visual) {
            (PreparedPrompts::Text(prompts), None) => prompts
                .managed_inputs(&recipe.config, &host)
                .map_err(ImageGenerationFailure::Preflight),
            (PreparedPrompts::Image(visual), Some(stage)) => {
                let span = self.trace_phase_started(
                    None,
                    Some(&recipe.geometry.identity),
                    crate::image_generation::continuous_trace::ImageTracePhase::ConditioningVision,
                );
                let stages = std::iter::once(stage)
                    .chain(&prepared.additional_visuals)
                    .collect::<Vec<_>>();
                let result = self.execute_vision_image_inputs(
                    visual,
                    &stages,
                    image.expect("checked admitted image"),
                    &recipe.config,
                    &recipe.geometry.identity,
                    job,
                    receipts,
                    reuse,
                );
                if let (Some(start), Some(trace), Ok((_, Some(binding)))) =
                    (span, self.execution.continuous_trace.as_ref(), &result)
                {
                    #[derive(serde::Serialize)]
                    struct Finished<'a> {
                        kind: &'static str,
                        start_sequence: u64,
                        outcome: &'static str,
                        main_vision_binding: &'a super::vision::reuse::MainVisionBindingReceipt,
                    }
                    trace.handle().record_diagnostic(&Finished {
                        kind: "phase_finished",
                        start_sequence: start,
                        outcome: "completed",
                        main_vision_binding: binding,
                    });
                } else {
                    self.trace_phase_finished(
                    span,
                    if result.is_ok() {
                        crate::image_generation::continuous_trace::ImageTracePhaseOutcome::Completed
                    } else {
                        crate::image_generation::continuous_trace::ImageTracePhaseOutcome::Failed
                    },
                    );
                }
                result.map(|(inputs, _)| inputs)
            }
            _ => unreachable!("presentation/stage presence checked before execution"),
        }
        .map_err(pipeline_failure)?;
        let encoder = self
            .execution
            .start_stage(prepared.encoder.clone(), job, receipts)
            .map_err(pipeline_failure)?;
        let (encoder, taps) = execute_component(&mut self.execution, encoder, job, inputs)
            .map_err(pipeline_failure)?;
        let normalized = match standardized_input(
            &host,
            &recipe.config,
            &encoder,
            &taps,
            &adapter,
            &checkpoint,
        ) {
            Ok(input) => input,
            Err(source) => return Err(pipeline_failure(encoder.processing(source))),
        };
        if let Err(source) = super::conditioning_diagnostics::observe_validated_main_text(
            &mut self.main_text_diagnostics,
            &recipe.geometry.identity,
            job,
            recipe.config.encoder().text().hidden_state_indices(),
            u64::from(recipe.config.encoder().text().layout().sequence_length()),
            &taps,
        ) {
            return Err(pipeline_failure(encoder.processing(source)));
        }
        // N owns its own Input bytes. Five readbacks retire before TE session
        // retirement and before the projection's installation or activation.
        drop(taps);
        self.execution
            .finish_stage(encoder, receipts)
            .map_err(pipeline_failure)?;

        let projection = self
            .execution
            .start_stage(prepared.connector.clone(), job, receipts)
            .map_err(pipeline_failure)?;
        let (projection, outputs) =
            execute_component(&mut self.execution, projection, job, vec![normalized])
                .map_err(pipeline_failure)?;
        let finish = || -> Result<_> {
            ensure!(
                outputs.len() == 1,
                "connector projection must return one declared output"
            );
            ensure!(
                host.owns(outputs[0].bytes())
                    && outputs[0].bytes().class() == HostMemoryClass::Output,
                "connector projection readback belongs to a foreign Host owner"
            );
            adapter.managed_conditioning(
                &host,
                &outputs[0],
                &checkpoint,
                PH3ImageConditioningPrecision::F32,
            )
        };
        let conditioning = match finish() {
            Ok(conditioning) => conditioning,
            Err(source) => return Err(pipeline_failure(projection.processing(source))),
        };
        drop(outputs);
        self.execution
            .finish_stage(projection, receipts)
            .map_err(pipeline_failure)?;
        Ok(conditioning)
    }
}

fn validate_connector_domain(
    selected: &PH3ImageTextConnectorConfig,
    source: &PH3ImageTextConnectorConfig,
) -> Result<()> {
    // Reconstructing the selected encoder taps omits serialization provenance
    // such as `_diffusers_version`. The original metadata digest retains that
    // provenance; every mathematical field must still match the source.
    ensure!(
        selected.hidden_state_indices() == source.hidden_state_indices()
            && selected.text_hidden_size() == source.text_hidden_size()
            && selected.out_channels() == source.out_channels(),
        "conditioning connector differs from the validated source domain"
    );
    Ok(())
}

fn standardized_input<R: DeviceRuntime>(
    host: &HostMemoryAccount,
    config: &PH3ImageTextConditioningConfig,
    encoder: &ActiveImageStage<R>,
    outputs: &[TensorExecutionOutput],
    adapter: &PH3ImageTextConnectorManagedAdapter<'_>,
    checkpoint: &PH3ImageTextConnectorCheckpoint<'_>,
) -> Result<ManagedHostTensor> {
    let ports = encoder
        .session
        .executable()
        .execution_plan()
        .payload()
        .tensor_ports()
        .context("text encoder lacks declared tensor ports")?;
    ensure!(
        outputs.len() == config.encoder().text().hidden_state_indices().len(),
        "text encoder must read back exactly its selected F32 taps"
    );
    let states = config
        .encoder()
        .text()
        .hidden_state_indices()
        .iter()
        .map(|&index| {
            let id = Qwen3VLTextComfyFamily::hidden_state_output_id(index)?;
            let port = ports
                .outputs()
                .iter()
                .find(|p| p.value_id() == &id)
                .context("encoder lacks the selected declared tap port")?;
            let output = outputs
                .iter()
                .find(|o| o.value_id() == &id)
                .context("encoder did not return the selected actual tap")?;
            ensure!(
                host.owns(output.bytes()) && output.bytes().class() == HostMemoryClass::Output,
                "encoder tap readback belongs to a foreign Host owner"
            );
            PH3ImageF32HiddenState::from_output(u32::try_from(index)?, port, output)
        })
        .collect::<Result<Vec<_>>>()?;
    adapter.managed_standardized_f32_input(host, &states, checkpoint)
}

/// Both text components use this existing ActiveImageStage execution path.
/// Their separate real calls record distinct attempts, completions and device
/// profiles. A failed operation keeps its actual active session in the source.
pub(super) fn execute_component<R: DeviceRuntime, S: ImagePipelineSourceAdapter>(
    execution: &mut ImagePipelineExecution<R, S>,
    mut active: ActiveImageStage<R>,
    job: u64,
    inputs: Vec<ManagedHostTensor>,
) -> std::result::Result<(ActiveImageStage<R>, Vec<TensorExecutionOutput>), ImageGenerationFailure<R>>
{
    if let Err(source) = active
        .session
        .set_output_host_account(execution.execution_host().clone())
    {
        return Err(active.processing(source.into()));
    }
    let Some(actor) = execution.serial_actor.as_mut() else {
        return Err(active.processing(anyhow::anyhow!("image serial actor is closed")));
    };
    active.execute_on_actor(actor, job, inputs)
}

fn pipeline_failure<R: DeviceRuntime>(
    source: ImageGenerationFailure<R>,
) -> ImageGenerationFailure<R> {
    // There is at most one active text component: the inner Execution or
    // Processing failure already owns it. The caller appends its persistent
    // scopes/pins, while completed stages remain in the ordinary receipts.
    ImageGenerationFailure::Pipeline {
        source: Box::new(source),
        stages: vec![],
        weights: vec![],
        host_scopes: vec![],
        host_backing: vec![],
    }
}

#[cfg(test)]
mod tests;
