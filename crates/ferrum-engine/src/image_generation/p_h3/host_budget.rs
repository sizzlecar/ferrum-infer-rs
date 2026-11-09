//! Already-physical Host requirements for the existing H3 consumers. This
//! adapter allocates no tensors and retains no account or request scope.
use super::{
    stages::{CompiledConditioning, CompiledSteady, PreparedGeneration},
    PH3ImagePipeline,
};
use crate::image_generation::{
    host_budget::{self, ExecutionHostBudget},
    PreparedImageStage,
};
use anyhow::{ensure, Context, Result};
use ferrum_models::{
    normal_noise::NormalNoiseConfig,
    p_h3_image::{
        PH3ImageConditioningPrecision, PH3ImageRefinementHostRequirements,
        PH3ImageTextConditioningHostRequirements,
    },
    p_h3_image_vae::PH3ImageVaeDecoderHostRequirements,
};
use ferrum_runtime::*;

pub(super) struct GenerationHostRequirements {
    /// Two separately padded successor slots for each image/audio sample,
    /// plus the separately owned B1 refinement cache. No decoder canvas is
    /// reserved here: its actual consumer uses one combined Output account.
    pub persistent_input_bytes: usize,
    pub initialization: ExecutionHostBudget,
    pub steady: ExecutionHostBudget,
    /// Independent stage scope, including the canvas once. Closing it after
    /// finish returns unused work credit; pixels retain their real charge.
    pub decoder: ExecutionHostBudget,
    pub conditioning: PH3ImageTextConditioningHostRequirements,
    pub visual: Option<VisionHostRequirements>,
    pub refinement: PH3ImageRefinementHostRequirements,
    pub decoder_detail: PH3ImageVaeDecoderHostRequirements,
    pub initialization_backing: Vec<HostBackingPreparation>,
    pub decoder_backing: Vec<HostBackingPreparation>,
}

/// Per-stage newly allocated payloads. The original decoded owner is borrowed
/// through initialization and remains the edit caller's separate ingress grant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct VisionHostRequirements {
    pub image_input_bytes: usize,
    pub visual_output_bytes: usize,
    pub text_input_bytes: usize,
    pub input_peak_bytes: usize,
    pub output_peak_bytes: usize,
    pub live_peak_bytes: usize,
}
impl VisionHostRequirements {
    fn from_physical(image_input: usize, visual_output: usize, text_input: usize) -> Result<Self> {
        // Inputs of the visual execution retire when its readbacks return.
        // The four Output owners then coexist only with the seven TE Inputs.
        let visual = image_input
            .checked_add(visual_output)
            .context("visual live Host backing overflows")?;
        let transfer = text_input
            .checked_add(visual_output)
            .context("visual-to-text live Host backing overflows")?;
        Ok(Self {
            image_input_bytes: image_input,
            visual_output_bytes: visual_output,
            text_input_bytes: text_input,
            input_peak_bytes: image_input.max(text_input),
            output_peak_bytes: visual_output,
            live_peak_bytes: visual.max(transfer),
        })
    }
}
impl GenerationHostRequirements {
    /// One conservative grant for the first complete B1 consumer. Samples,
    /// successors and the refinement cache remain charged while each stage
    /// uses its transient peak. Decoder input excludes the borrowed sample;
    /// its complete Output peak already includes the final canvas once.
    pub fn whole_request_budget(&self) -> Result<ExecutionHostBudget> {
        Ok(ExecutionHostBudget {
            input: self
                .persistent_input_bytes
                .checked_add(
                    self.initialization
                        .input
                        .max(self.steady.input)
                        .max(self.decoder.input),
                )
                .context("H3 complete request Input backing overflows")?,
            output: self
                .initialization
                .output
                .max(self.steady.output)
                .max(self.decoder.output),
            staging: self.initialization.staging.max(self.decoder.staging),
            // Initialization and decoder can retain different materializer
            // cache owners. Summing their bounds also conservatively retains
            // both transient source allowances; neither is omitted or reused
            // as authority for a different class.
            weights: self
                .initialization
                .weights
                .checked_add(self.decoder.weights)
                .context("H3 complete request WeightBacking overflows")?,
        })
    }

    pub fn into_backing(mut self) -> Vec<HostBackingPreparation> {
        self.initialization_backing
            .append(&mut self.decoder_backing);
        self.initialization_backing
    }

    pub fn persistent_claims(&self) -> [(HostMemoryClass, usize); 1] {
        [(
            HostMemoryClass::InputPreprocess,
            self.persistent_input_bytes,
        )]
    }
}

impl<R: DeviceRuntime> PH3ImagePipeline<R> {
    pub(super) fn generation_host_requirements(
        &self,
        prepared: &PreparedGeneration<R>,
    ) -> Result<GenerationHostRequirements> {
        let host = self.execution.host_account();
        let conditioning = prepared
            .conditioning
            .recipe
            .config
            .host_requirements(host, PH3ImageConditioningPrecision::F32)?;
        let visual = self.visual_host_requirements(&prepared.conditioning)?;
        let refinement = prepared
            .denoising
            .refine
            .recipe
            .refinement_host_requirements(host)?;
        ensure!(
            refinement.persistent_cache_by_request.len() == 1
                && refinement.persistent_cache_by_request[0].request
                    == prepared.conditioning.recipe.geometry.identity,
            "B1 Host requirements differ from the prepared request/cache identity"
        );
        let image_audio = prepared.noise.layout.input_bytes(host)?;
        let persistent_input_bytes = image_audio
            .checked_mul(2)
            .and_then(|n| n.checked_add(refinement.persistent_cache_bytes))
            .and_then(|n| {
                n.checked_add(if self.execution.sources.assets.has_prompt_expansion() {
                    conditioning.retained_conditioning_bytes
                } else {
                    0
                })
            })
            .context("H3 persistent sample/cache backing overflows")?;
        let steady = self.steady_host_requirements(&prepared.denoising.steady)?;
        let (noise_input, noise_output) = match prepared.noise.stage() {
            Some(stage) => (
                host.backing_bytes(usize::try_from(
                    NormalNoiseConfig::state_tensor()?.minimum_storage_bytes()?,
                )?)?,
                output_bytes(host, stage)?,
            ),
            None => (prepared.noise.layout.cpu_work_bytes(host)?, 0),
        };
        // Final conditioning remains borrowed while the BF16 refinement input
        // is built. New cache destinations consume persistent, not work credit.
        let refine_input = conditioning
            .retained_conditioning_bytes
            .checked_add(refinement.work_input_bytes)
            .context("H3 refinement Input backing overflows")?;
        let mut installable = Vec::with_capacity(5);
        if let Some(stage) = &prepared.conditioning.visual {
            installable.push(stage);
        }
        installable.extend([
            &prepared.conditioning.encoder,
            &prepared.conditioning.connector,
        ]);
        installable.extend(prepared.noise.stage());
        installable.push(&prepared.denoising.install.stage);
        let (weights, initialization_backing) = self.source_backing_budget(&installable)?;
        let staging = self.static_staging_budget(&installable)?;
        let initialization = ExecutionHostBudget {
            input: conditioning
                .input_peak_bytes
                .max(visual.map_or(0, |v| v.input_peak_bytes))
                .max(refine_input)
                .max(noise_input)
                .max(steady.input),
            output: conditioning
                .output_peak_bytes
                .max(visual.map_or(0, |v| v.output_peak_bytes))
                .max(refinement.work_output_bytes)
                .max(noise_output)
                .max(steady.output),
            staging,
            weights,
        };
        let decoder_detail = prepared.decoder.recipe.host_requirements(host)?;
        let decoder_stages = prepared.decoder.stages.iter().collect::<Vec<_>>();
        let (decoder_weights, decoder_backing) = self.source_backing_budget(&decoder_stages)?;
        let decoder = ExecutionHostBudget {
            // The model query excludes the existing normalized sample owner.
            input: decoder_detail.input_peak_bytes,
            // Canvas, seams, readbacks and tiles share the consumer's Output
            // view. Include the full peak once, without a second canvas grant.
            output: decoder_detail.output_peak_bytes,
            staging: self.static_staging_budget(&decoder_stages)?,
            weights: decoder_weights,
        };
        Ok(GenerationHostRequirements {
            persistent_input_bytes,
            initialization,
            steady,
            decoder,
            conditioning,
            visual,
            refinement,
            decoder_detail,
            initialization_backing,
            decoder_backing,
        })
    }

    pub(super) fn visual_host_requirements(
        &self,
        prepared: &CompiledConditioning<R>,
    ) -> Result<Option<VisionHostRequirements>> {
        match (&prepared.visual, prepared.recipe.prompts.visual()) {
            (None, None) => Ok(None),
            (Some(stage), Some(recipe)) => {
                let host = self.execution.host_account();
                ensure!(
                    stage.family().fingerprint()? == recipe.family.fingerprint()?,
                    "visual budget belongs to another prepared recipe"
                );
                let visual_ports = host_budget::ports(stage)?;
                let text_ports = host_budget::ports(&prepared.encoder)?;
                let image_input = host_budget::sum_ports(host, visual_ports.inputs().iter())?;
                let visual_output = host_budget::sum_ports(host, visual_ports.outputs().iter())?;
                let text_input = host_budget::sum_ports(host, text_ports.inputs().iter())?;
                Ok(Some(VisionHostRequirements::from_physical(
                    image_input,
                    visual_output,
                    text_input,
                )?))
            }
            _ => {
                anyhow::bail!("visual budget requires the matching presentation and compiled stage")
            }
        }
    }

    /// New steady packing buffers/readbacks only. Existing request sample and
    /// refinement-cache owners, including both Euler successors, stay in each
    /// request's persistent scope and are never reserved here a second time.
    pub(super) fn steady_host_requirements(
        &self,
        prepared: &CompiledSteady<R>,
    ) -> Result<ExecutionHostBudget> {
        let host = self.execution.host_account();
        let ports = host_budget::ports(&prepared.stage)?;
        Ok(ExecutionHostBudget {
            input: host_budget::sum_ports(host, ports.inputs().iter())?,
            output: host_budget::sum_ports(host, ports.outputs().iter())?,
            staging: 0,
            weights: 0, // RequireShared forbids any source installation.
        })
    }

    pub(super) fn source_backing_budget(
        &self,
        stages: &[&PreparedImageStage<R>],
    ) -> Result<(usize, Vec<HostBackingPreparation>)> {
        let mut cache = 0usize;
        let mut transient = 0usize;
        let mut retained = Vec::with_capacity(stages.len());
        for stage in stages {
            let (total, preparation) = self.execution.stage_host_backing(stage)?;
            // Real persistent Host cache fills accumulate across stages.
            // Source payload conversions retire after their installation,
            // so their independently reported transient peaks take a max.
            cache = cache
                .checked_add(preparation.additional_bytes())
                .context("H3 source cache backing overflows")?;
            transient = transient.max(
                total
                    .checked_sub(preparation.additional_bytes())
                    .context("source bound excludes its materializer backing")?,
            );
            retained.push(preparation);
        }
        Ok((
            cache
                .checked_add(transient)
                .context("H3 source Host backing overflows")?,
            retained,
        ))
    }
    pub(super) fn static_staging_budget(&self, stages: &[&PreparedImageStage<R>]) -> Result<usize> {
        if !self.execution.runtime.requires_managed_static_staging() {
            return Ok(0);
        }
        stages.iter().try_fold(0usize, |peak, stage| -> Result<_> {
            Ok(peak.max(host_budget::staging_bytes(
                self.execution.host_account(),
                stage,
                self.execution.config.static_initialization,
            )?))
        })
    }
}
fn output_bytes<R: DeviceRuntime>(
    host: &HostMemoryAccount,
    stage: &PreparedImageStage<R>,
) -> Result<usize> {
    Ok(host_budget::sum_ports(
        host,
        host_budget::ports(stage)?.outputs().iter(),
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn visual_handoff_uses_coexisting_owners_without_summing_disjoint_phase_peaks() {
        // Values are already separately padded physical credits. Borrowing an
        // admitted decoded owner is not a new allocation in either phase.
        let b = VisionHostRequirements::from_physical(320, 160, 704).unwrap();
        assert_eq!(b.input_peak_bytes, 704);
        assert_eq!(b.output_peak_bytes, 160);
        assert_eq!(b.live_peak_bytes, 864);
        let larger_image = VisionHostRequirements::from_physical(1024, 160, 704).unwrap();
        assert_eq!(larger_image.live_peak_bytes, 1184);
        assert_eq!(larger_image.input_peak_bytes, 1024);
        assert!(VisionHostRequirements::from_physical(usize::MAX, 1, 0).is_err());
        assert!(VisionHostRequirements::from_physical(0, 1, usize::MAX).is_err());
    }
}
