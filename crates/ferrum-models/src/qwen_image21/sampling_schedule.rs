//! Metadata-selected deterministic Euler schedule for the Qwen-Image 2.1 adapter.
//!
//! Qwen-Image 2.1 uses explicit model sigmas or a resolution-dependent schedule
//! when the scheduler declares dynamic shifting and the model supplies no sigmas.
//! Both are converted to FP32 before `sigma * 1000` and end with terminal zero.
//!
//! Source contract: vLLM-Omni c0551d22674d0db81305c800b66ad83035fdc980
//! `pipeline_qwen_image_21.py` prepare_timesteps, and diffusers v0.40.0
//! `scheduling_flow_match_euler_discrete.py` set_timesteps. Parsing this contract
//! is not proof of deployed Torch trajectory parity or metadata authenticity.

use anyhow::{ensure, Context, Result};
use ferrum_diffusion::{EulerStep, FlowMatchEulerSchedule};
use serde::{de, Deserialize, Deserializer};
use std::fmt;

/// Public adapter capacity for explicit evaluations. The endpoint is separate
/// and does not consume an evaluation. The checked target metadata uses eight;
/// any valid list of 1..=256 entries is supported by this adapter.
pub const QWEN_IMAGE21_MAX_SAMPLING_EVALUATIONS: usize = 256;

const SCHEDULER_CLASS: &str = "FlowMatchEulerDiscreteScheduler";
const TIMESTEP_SCALE: f32 = 1000.0;
const TERMINAL_SIGMA: f32 = 0.0;
const QWEN_IMAGE21_DEFAULT_EVALUATIONS: usize = 40;

// Ignore unrelated pipeline component declarations, but deserialize consumed
// fields directly: derived struct parsing rejects duplicate scheduler or sigma
// fields before a Value map could silently replace an earlier occurrence.
#[derive(Deserialize)]
struct ModelIndex {
    scheduler: [String; 2],
    sample_sigmas: Option<ExplicitSigmas>,
}

struct ExplicitSigmas(Vec<f32>);

impl<'de> Deserialize<'de> for ExplicitSigmas {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> de::Visitor<'de> for Visitor {
            type Value = ExplicitSigmas;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(
                    f,
                    "1..={QWEN_IMAGE21_MAX_SAMPLING_EVALUATIONS} numeric sigmas"
                )
            }

            fn visit_seq<A: de::SeqAccess<'de>>(
                self,
                mut sequence: A,
            ) -> Result<Self::Value, A::Error> {
                let mut sigmas = Vec::new();
                while let Some(value) = sequence.next_element::<f64>()? {
                    if sigmas.len() == QWEN_IMAGE21_MAX_SAMPLING_EVALUATIONS {
                        return Err(de::Error::custom(
                            "explicit sigma list exceeds adapter capacity",
                        ));
                    }
                    // Match the upstream explicit schedule's FP32 entry point.
                    // F32 overflow, zero underflow and monotonic collapse are
                    // subsequently rejected by FlowMatchEulerSchedule.
                    sigmas.push(value as f32);
                }
                if sigmas.is_empty() {
                    return Err(de::Error::custom("explicit sigma list must be nonempty"));
                }
                Ok(ExplicitSigmas(sigmas))
            }
        }
        deserializer.deserialize_seq(Visitor)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Scheduler {
    #[serde(rename = "_class_name")]
    class: String,
    #[serde(rename = "_diffusers_version")]
    version: String,
    base_image_seq_len: u64,
    base_shift: f64,
    invert_sigmas: bool,
    max_image_seq_len: u64,
    max_shift: f64,
    num_train_timesteps: u32,
    shift: f64,
    shift_terminal: Option<f64>,
    stochastic_sampling: bool,
    time_shift_type: String,
    use_beta_sigmas: bool,
    use_dynamic_shifting: bool,
    use_exponential_sigmas: bool,
    use_karras_sigmas: bool,
}

/// Immutable schedule plus informational producer metadata. No request progress
/// is stored here: each ManagedBf16Euler receives its own clone of `solver()`.
#[derive(Clone, Debug)]
pub struct QwenImage21SamplingSchedule {
    solver: FlowMatchEulerSchedule,
    metadata_diffusers_version: String,
}

/// A validated model schedule, resolved against the target image's latent extent.
pub struct QwenImage21SamplingPlan(SamplingKind);

enum SamplingKind {
    Explicit(QwenImage21SamplingSchedule),
    Dynamic(Scheduler),
}

impl QwenImage21SamplingPlan {
    pub fn parse(model_index: &[u8], scheduler: &[u8]) -> Result<Self> {
        let (index, config) = parse_metadata(model_index, scheduler)?;
        if config.use_dynamic_shifting {
            ensure!(
                index.sample_sigmas.is_none(),
                "the dynamic schedule requires a model without explicit sigmas"
            );
            ensure!(
                config.max_image_seq_len > config.base_image_seq_len,
                "the dynamic schedule requires two distinct sequence lengths"
            );
            if let Some(terminal) = config.shift_terminal {
                ensure!(
                    terminal > 0.0 && terminal < 1.0,
                    "the last evaluation sigma must be between zero and one"
                );
            }
            Ok(Self(SamplingKind::Dynamic(config)))
        } else {
            Ok(Self(SamplingKind::Explicit(
                QwenImage21SamplingSchedule::from_metadata(index, config)?,
            )))
        }
    }

    pub fn for_image_latents(&self, image_seq_len: usize) -> Result<FlowMatchEulerSchedule> {
        match &self.0 {
            SamplingKind::Explicit(schedule) => Ok(schedule.solver().clone()),
            SamplingKind::Dynamic(config) => {
                ensure!(
                    image_seq_len > 0,
                    "the image latent sequence must be nonempty"
                );
                // Diffusers interpolates the shift against the target latent
                // sequence, applies it in FP32, and optionally stretches the
                // evaluations before appending the zero endpoint.
                let slope = (config.max_shift - config.base_shift)
                    / (config.max_image_seq_len - config.base_image_seq_len) as f64;
                let intercept = config.base_shift - slope * config.base_image_seq_len as f64;
                let shift = (image_seq_len as f64 * slope + intercept).exp() as f32;
                ensure!(
                    shift.is_finite() && shift > 0.0,
                    "invalid shift for the requested image latent extent"
                );
                let mut sigmas = [0.0; QWEN_IMAGE21_DEFAULT_EVALUATIONS];
                for (index, value) in sigmas.iter_mut().enumerate() {
                    let sigma = ((QWEN_IMAGE21_DEFAULT_EVALUATIONS - index) as f64
                        / QWEN_IMAGE21_DEFAULT_EVALUATIONS as f64)
                        as f32;
                    *value = shift / (shift + (1.0 / sigma - 1.0));
                }
                if let Some(terminal) = config.shift_terminal {
                    let scale = (1.0 - sigmas[QWEN_IMAGE21_DEFAULT_EVALUATIONS - 1])
                        / (1.0 - terminal as f32);
                    for sigma in &mut sigmas {
                        *sigma = 1.0 - (1.0 - *sigma) / scale;
                    }
                }
                Ok(FlowMatchEulerSchedule::new(
                    &sigmas,
                    TERMINAL_SIGMA,
                    TIMESTEP_SCALE,
                )?)
            }
        }
    }

    pub fn evaluation_count(&self) -> usize {
        match &self.0 {
            SamplingKind::Explicit(schedule) => schedule.evaluation_count(),
            SamplingKind::Dynamic(_) => QWEN_IMAGE21_DEFAULT_EVALUATIONS,
        }
    }
}

impl QwenImage21SamplingSchedule {
    /// Consume the original JSON bytes, not a pre-collapsed JSON object, so
    /// duplicate consumed fields are rejected. Unknown scheduler fields are
    /// rejected; `shift_terminal` may be absent or null (both disable shifting).
    /// Unrelated model-index component fields are ignored. This API performs no
    /// I/O; the source loader remains responsible for metadata byte/read limits.
    pub fn parse(model_index: &[u8], scheduler: &[u8]) -> Result<Self> {
        let (index, config) = parse_metadata(model_index, scheduler)?;
        Self::from_metadata(index, config)
    }

    fn from_metadata(index: ModelIndex, config: Scheduler) -> Result<Self> {
        ensure!(
            !config.use_dynamic_shifting && config.shift_terminal.is_none(),
            "the explicit sigma schedule must not request dynamic or terminal shifting"
        );
        let sigmas = index
            .sample_sigmas
            .context("missing explicit sample_sigmas")?;
        Ok(Self {
            solver: FlowMatchEulerSchedule::new(&sigmas.0, TERMINAL_SIGMA, TIMESTEP_SCALE)?,
            metadata_diffusers_version: config.version,
        })
    }

    pub fn solver(&self) -> &FlowMatchEulerSchedule {
        &self.solver
    }

    pub fn into_solver(self) -> FlowMatchEulerSchedule {
        self.solver
    }

    pub fn steps(&self) -> &[EulerStep] {
        self.solver.steps()
    }

    pub fn evaluation_count(&self) -> usize {
        self.steps().len()
    }

    /// Informational producer version, not a capability gate or authentication.
    pub fn metadata_diffusers_version(&self) -> &str {
        &self.metadata_diffusers_version
    }

    pub fn terminal_sigma(&self) -> f32 {
        TERMINAL_SIGMA
    }

    pub fn timestep_scale(&self) -> f32 {
        TIMESTEP_SCALE
    }
}

fn parse_metadata(model_index: &[u8], scheduler: &[u8]) -> Result<(ModelIndex, Scheduler)> {
    // Require JSON objects while preserving duplicate-field checks.
    for (name, bytes) in [("model index", model_index), ("scheduler", scheduler)] {
        ensure!(
            bytes.iter().copied().find(|b| !b.is_ascii_whitespace()) == Some(b'{'),
            "{name} metadata must be a JSON object"
        );
    }
    let index: ModelIndex = serde_json::from_slice(model_index)?;
    ensure!(
        index.scheduler[0] == "diffusers" && index.scheduler[1] == SCHEDULER_CLASS,
        "model index must declare the supported FlowMatchEuler scheduler"
    );
    let config: Scheduler = serde_json::from_slice(scheduler)?;
    ensure!(
        config.class == SCHEDULER_CLASS && config.num_train_timesteps == 1000,
        "scheduler type or time units differ from the checked timestep adapter"
    );
    ensure!(
        !config.invert_sigmas
            && !config.stochastic_sampling
            && !config.use_beta_sigmas
            && !config.use_exponential_sigmas
            && !config.use_karras_sigmas
            && config.shift == 1.0
            && config.time_shift_type == "exponential",
        "unsupported FlowMatchEuler scheduler options"
    );
    ensure!(
        config.base_image_seq_len > 0
            && config.max_image_seq_len >= config.base_image_seq_len
            && config.base_shift.is_finite()
            && config.max_shift.is_finite(),
        "invalid scheduler geometry"
    );
    Ok((index, config))
}

#[cfg(test)]
mod tests;
