//! The deterministic P-H3 schedule, following Diffusers
//! `c55ec27943462eeeb65c177d6ff7d8272d21d7a7` scheduling_minimax_h3.py.
//!
//! Explicit sigma grids include the terminal zero. Model time is `1 - sigma`,
//! the model predicts a data-ward velocity, and the reference retains the
//! round trip `1 - timestep` separately from the grid sigma in its Euler blend.
//! This module implements the FP32 path used by P-H3's image/audio latent rows;
//! it does not claim Torch/CUDA trajectory parity or implement stochastic H3.

use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use ferrum_diffusion::FlowMatchEulerSchedule;
use serde::Deserialize;

use super::config::ensure_object;

mod managed;
pub use managed::{
    PH3ImageManagedSampling, PH3ImageSamplingVelocity, PH3ImageVelocityBatchLayout,
    PH3ImageVelocityPrecision, PH3ImageVelocityRequest, PreparedPH3ImageSamplingAdvance,
};

const SCHEDULER_CLASS: &str = "MiniMaxH3Scheduler";

/// Caller-declared metadata and retained schedule capacity. These are resource
/// limits, not a model-specific step count: the released grid has eight
/// evaluations, while other declared explicit grids may use different counts.
#[derive(Clone, Copy, Debug)]
pub struct PH3ImageSamplingLimits {
    pub max_metadata_bytes: usize,
    pub max_evaluations: usize,
}

#[derive(Deserialize)]
struct ModelIndex {
    #[serde(rename = "_class_name")]
    class: String,
    #[serde(rename = "_blocks_class_name")]
    blocks_class: String,
    scheduler: ComponentDeclaration,
    audio_scheduler: ComponentDeclaration,
    sample_sigmas: Vec<f32>,
}

// The third entry describes source registration, which this schedule parser
// neither resolves nor authorizes. The pipeline loader must validate it before
// loading components. Preserve direct parsing to detect duplicate used fields.
#[derive(Deserialize)]
struct ComponentDeclaration(String, String, serde::de::IgnoredAny);

impl ComponentDeclaration {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.0 == "diffusers" && self.1 == SCHEDULER_CLASS,
            "P-H3-Image must declare MiniMaxH3Scheduler for both latent streams"
        );
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SchedulerConfig {
    #[serde(rename = "_class_name")]
    class: String,
    #[serde(rename = "_diffusers_version")]
    _version: Option<String>,
    shift: f64,
    shift_provided_sigmas: bool,
    #[serde(default)]
    stochastic_sampling: bool,
}

impl SchedulerConfig {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.class == SCHEDULER_CLASS,
            "unsupported P-H3-Image scheduler class"
        );
        ensure!(
            self.shift.is_finite() && self.shift > 0.0,
            "scheduler shift must be finite and positive"
        );
        // The released checkpoint declares a finished grid. Retain its shift
        // as metadata but never apply that shift to the grid a second time.
        ensure!(
            !self.shift_provided_sigmas && !self.stochastic_sampling,
            "P-H3-Image supports explicit verbatim deterministic sigmas only"
        );
        Ok(())
    }
}

/// Immutable model schedule. Each request starts its own cursor; no global
/// scheduler index can be shared accidentally by concurrently active requests.
#[derive(Clone, Debug)]
pub struct PH3ImageSamplingSchedule {
    steps: Arc<[PH3ImageSamplingStep]>,
}

impl PH3ImageSamplingSchedule {
    pub fn parse(
        model_index: &[u8],
        scheduler: &[u8],
        audio_scheduler: &[u8],
        limits: PH3ImageSamplingLimits,
    ) -> Result<Self> {
        ensure!(
            limits.max_metadata_bytes > 0 && limits.max_evaluations > 0,
            "P-H3-Image sampling capacities must be positive"
        );
        for bytes in [model_index, scheduler, audio_scheduler] {
            ensure!(
                bytes.len() <= limits.max_metadata_bytes,
                "P-H3-Image sampling metadata exceeds byte capacity"
            );
            ensure_object(bytes)?;
        }
        let index: ModelIndex = serde_json::from_slice(model_index)
            .context("invalid P-H3-Image modular model index")?;
        ensure!(
            index.class == "PH3ImageModularPipeline" && index.blocks_class == "PH3ImageBlocks",
            "unsupported P-H3-Image pipeline declaration"
        );
        index.scheduler.validate()?;
        index.audio_scheduler.validate()?;
        for bytes in [scheduler, audio_scheduler] {
            let config: SchedulerConfig =
                serde_json::from_slice(bytes).context("invalid P-H3-Image scheduler metadata")?;
            config.validate()?;
        }
        ensure!(
            index.sample_sigmas.len() >= 2,
            "P-H3-Image sigma grid needs an evaluation and terminal zero"
        );
        ensure!(
            index.sample_sigmas.len() - 1 <= limits.max_evaluations,
            "P-H3-Image sigma grid exceeds evaluation capacity"
        );
        let (&terminal, evaluations) = index.sample_sigmas.split_last().unwrap();
        ensure!(terminal == 0.0, "P-H3-Image sigma grid must end with zero");
        // Reuse only the shared finite, decreasing grid validation. H3 model
        // time and velocity integration below deliberately use different math.
        let grid = FlowMatchEulerSchedule::new(evaluations, terminal, 1.0)?;
        Ok(Self {
            steps: grid
                .steps()
                .iter()
                .map(|step| PH3ImageSamplingStep {
                    index: step.index(),
                    sigma: step.sigma(),
                    next_sigma: step.next_sigma(),
                    timestep: 1.0_f32 - step.sigma(),
                })
                .collect(),
        })
    }

    pub fn steps(&self) -> &[PH3ImageSamplingStep] {
        &self.steps
    }
    pub fn evaluation_count(&self) -> usize {
        self.steps.len()
    }
    pub fn start(&self) -> PH3ImageRequestSchedule {
        PH3ImageRequestSchedule {
            schedule: self.clone(),
            step_index: 0,
        }
    }
}

/// One H3 model evaluation. The terminal sigma is an integration endpoint,
/// never an additional transformer call.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PH3ImageSamplingStep {
    index: usize,
    sigma: f32,
    next_sigma: f32,
    timestep: f32,
}

impl PH3ImageSamplingStep {
    pub fn index(self) -> usize {
        self.index
    }
    pub fn sigma(self) -> f32 {
        self.sigma
    }
    pub fn next_sigma(self) -> f32 {
        self.next_sigma
    }
    pub fn timestep(self) -> f32 {
        self.timestep
    }

    fn update(self, sample: f32, velocity: f32) -> f32 {
        let sigma_from_timestep = 1.0_f32 - self.timestep;
        let denoised = sample + sigma_from_timestep * velocity;
        let ratio = self.next_sigma / self.sigma;
        ratio * sample + (1.0_f32 - ratio) * denoised
    }
}

/// Request-local progress over the image and internal audio streams. P-H3
/// jointly denoises both, then discards the audio; omitting it changes the model.
#[derive(Debug)]
pub struct PH3ImageRequestSchedule {
    schedule: PH3ImageSamplingSchedule,
    step_index: usize,
}

impl PH3ImageRequestSchedule {
    pub fn step_index(&self) -> usize {
        self.step_index
    }
    pub fn is_complete(&self) -> bool {
        self.step_index == self.schedule.steps.len()
    }
    pub fn current_step(&self) -> Option<PH3ImageSamplingStep> {
        self.schedule.steps.get(self.step_index).copied()
    }

    /// Advance both streams in FP32 using the original H3 operation order.
    ///
    /// All lengths, values and results are checked before either stream or the
    /// cursor changes. Slice lengths describe the local numerical reference;
    /// device execution additionally needs checked shapes/layouts and runtime
    /// resource ownership. No tensor allocation or hidden RNG occurs here.
    pub fn advance_f32(
        &mut self,
        image: &mut [f32],
        image_velocity: &[f32],
        audio: &mut [f32],
        audio_velocity: &[f32],
    ) -> Result<PH3ImageSamplingStep> {
        let step = self
            .current_step()
            .context("P-H3-Image sampling is already complete")?;
        for (name, sample, velocity) in [
            ("image", &*image, image_velocity),
            ("audio", &*audio, audio_velocity),
        ] {
            ensure!(
                !sample.is_empty() && sample.len() == velocity.len(),
                "{name} sample and velocity must have equal, nonzero element counts"
            );
            for (&value, &prediction) in sample.iter().zip(velocity) {
                ensure!(
                    value.is_finite() && prediction.is_finite(),
                    "{name} sample and velocity must be finite"
                );
                ensure!(
                    step.update(value, prediction).is_finite(),
                    "{name} H3 update is not finite"
                );
            }
        }
        for (sample, velocity) in [(image, image_velocity), (audio, audio_velocity)] {
            for (value, &prediction) in sample.iter_mut().zip(velocity) {
                *value = step.update(*value, prediction);
            }
        }
        self.step_index += 1;
        Ok(step)
    }
}

#[cfg(test)]
mod tests;
