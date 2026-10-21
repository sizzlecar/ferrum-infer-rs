use std::sync::Arc;

use thiserror::Error;

/// The scheduler's FP32 time coordinate, distinct from a step index.
///
/// Model adapters own any normalization or conversion to a denoiser's dtype.
/// In particular, casting this value to BF16 and then normalizing it is not
/// necessarily equivalent to passing the original sigma directly to a model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SchedulerTimestep(f32);

impl SchedulerTimestep {
    pub fn as_f32(self) -> f32 {
        self.0
    }
}

/// One validated deterministic Euler interval and its evaluation time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EulerStep {
    index: usize,
    sigma: f32,
    next_sigma: f32,
    timestep: SchedulerTimestep,
}

impl EulerStep {
    pub fn index(self) -> usize {
        self.index
    }

    pub fn sigma(self) -> f32 {
        self.sigma
    }

    pub fn next_sigma(self) -> f32 {
        self.next_sigma
    }

    pub fn timestep(self) -> SchedulerTimestep {
        self.timestep
    }

    /// Negative for every supported denoising interval.
    pub fn delta_sigma(self) -> f32 {
        self.next_sigma - self.sigma
    }
}

/// An immutable explicit sigma schedule for deterministic flow-matching Euler.
///
/// Each supplied sigma denotes one denoiser evaluation. The terminal sigma is
/// supplied separately and adds an integration endpoint, not another model
/// evaluation. Timesteps are `sigma * timestep_scale`, computed in FP32.
///
/// The supported contract is finite, strictly decreasing sigmas in `[0, 1]`.
/// No shifting, inversion, schedule generation, stochastic sampling, or
/// per-token timesteps are inferred. A model adapter must resolve those semantics
/// before constructing this schedule; unsupported configurations must be rejected
/// by that adapter. Equal sigmas and FP32-collapsed timesteps are rejected rather
/// than resolved through floating-point timestep searches.
#[derive(Clone, Debug)]
pub struct FlowMatchEulerSchedule {
    steps: Arc<[EulerStep]>,
}

impl FlowMatchEulerSchedule {
    pub fn new(
        sigmas: &[f32],
        terminal_sigma: f32,
        timestep_scale: f32,
    ) -> Result<Self, ScheduleError> {
        if sigmas.is_empty() {
            return Err(ScheduleError::EmptySchedule);
        }
        if !timestep_scale.is_finite() || timestep_scale <= 0.0 {
            return Err(ScheduleError::InvalidTimestepScale);
        }
        if !valid_sigma(terminal_sigma) {
            return Err(ScheduleError::InvalidTerminalSigma);
        }
        let mut steps = Vec::with_capacity(sigmas.len());
        let mut previous_timestep = None;
        for (index, &sigma) in sigmas.iter().enumerate() {
            if !valid_sigma(sigma) {
                return Err(ScheduleError::InvalidSigma { index });
            }
            let next_sigma = sigmas.get(index + 1).copied().unwrap_or(terminal_sigma);
            if !valid_sigma(next_sigma) {
                return Err(ScheduleError::InvalidSigma { index: index + 1 });
            }
            if next_sigma >= sigma {
                return Err(ScheduleError::NonDecreasingInterval { index });
            }
            let timestep = sigma * timestep_scale;
            if !timestep.is_finite()
                || timestep <= 0.0
                || previous_timestep.is_some_and(|previous| timestep >= previous)
            {
                return Err(ScheduleError::UnrepresentableTimestep { index });
            }
            steps.push(EulerStep {
                index,
                sigma,
                next_sigma,
                timestep: SchedulerTimestep(timestep),
            });
            previous_timestep = Some(timestep);
        }
        Ok(Self {
            steps: steps.into(),
        })
    }

    pub fn steps(&self) -> &[EulerStep] {
        &self.steps
    }

    /// Create independent request progress over the same immutable schedule.
    pub fn start(&self) -> FlowMatchEuler {
        FlowMatchEuler {
            schedule: self.clone(),
            step_index: 0,
        }
    }
}

fn valid_sigma(sigma: f32) -> bool {
    sigma.is_finite() && (0.0..=1.0).contains(&sigma)
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ScheduleError {
    #[error("an explicit sigma schedule must contain at least one evaluation")]
    EmptySchedule,
    #[error("timestep scale must be finite and positive")]
    InvalidTimestepScale,
    #[error("terminal sigma must be finite and within [0, 1]")]
    InvalidTerminalSigma,
    #[error("sigma at index {index} must be finite and within [0, 1]")]
    InvalidSigma { index: usize },
    #[error("sigma interval {index} must be strictly decreasing, including the terminal endpoint")]
    NonDecreasingInterval { index: usize },
    #[error("timestep at index {index} must remain positive and strictly decreasing in FP32")]
    UnrepresentableTimestep { index: usize },
}

/// Request-local progress for a deterministic, FP32-only CPU Euler reference.
///
/// There is no shared mutable scheduler state and no hidden RNG. Callers supply
/// the initial latent and the velocity prediction for each evaluation. Flattened
/// slices must represent matching tensor shapes; this local numerical contract
/// validates element counts, while device execution must additionally validate
/// shape, layout, dtype, ownership, and runtime identity through the runtime.
#[derive(Debug)]
pub struct FlowMatchEuler {
    schedule: FlowMatchEulerSchedule,
    step_index: usize,
}

impl FlowMatchEuler {
    pub fn step_index(&self) -> usize {
        self.step_index
    }

    pub fn current_step(&self) -> Option<EulerStep> {
        self.schedule.steps.get(self.step_index).copied()
    }

    pub fn is_complete(&self) -> bool {
        self.step_index == self.schedule.steps.len()
    }

    /// Apply `sample + (next_sigma - sigma) * model_output` in FP32.
    ///
    /// Multiplication and addition are separate operations; this reference does
    /// not use fused multiply-add. Input and result validation precedes mutation,
    /// so an error leaves the entire sample and request progress unchanged.
    /// The second pass intentionally avoids allocating an unbounded scratch Vec.
    ///
    /// This covers the FP32 branch of deterministic Euler, not PyTorch's BF16
    /// scalar-promotion/intermediate-rounding or final output-cast behavior.
    pub fn advance_f32(
        &mut self,
        sample: &mut [f32],
        model_output: &[f32],
    ) -> Result<EulerStep, EulerStepError> {
        let step = self.current_step().ok_or(EulerStepError::Complete)?;
        if sample.len() != model_output.len() {
            return Err(EulerStepError::ElementCountMismatch {
                sample: sample.len(),
                model_output: model_output.len(),
            });
        }
        if sample.is_empty() {
            return Err(EulerStepError::EmptySample);
        }
        let delta = step.delta_sigma();
        for (index, (&value, &velocity)) in sample.iter().zip(model_output).enumerate() {
            if !value.is_finite() {
                return Err(EulerStepError::NonFiniteSample { index });
            }
            if !velocity.is_finite() {
                return Err(EulerStepError::NonFiniteModelOutput { index });
            }
            let increment = delta * velocity;
            let updated = value + increment;
            if !updated.is_finite() {
                return Err(EulerStepError::NonFiniteResult { index });
            }
        }
        for (value, &velocity) in sample.iter_mut().zip(model_output) {
            let increment = delta * velocity;
            *value += increment;
        }
        self.step_index += 1;
        Ok(step)
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum EulerStepError {
    #[error("the Euler schedule is already complete")]
    Complete,
    #[error("a latent sample must contain at least one element")]
    EmptySample,
    #[error("sample has {sample} elements but model output has {model_output}")]
    ElementCountMismatch { sample: usize, model_output: usize },
    #[error("sample element {index} is not finite")]
    NonFiniteSample { index: usize },
    #[error("model output element {index} is not finite")]
    NonFiniteModelOutput { index: usize },
    #[error("Euler result element {index} is not finite")]
    NonFiniteResult { index: usize },
}

#[cfg(test)]
mod tests;
