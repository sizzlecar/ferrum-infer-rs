use std::ops::Range;

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

use super::{FastH3Checkpoint, FAST_H3_GRID_MAX_T};

/// Reconstructible sampling identity shared by conditioning clocks and PDD
/// output heads. Derived coefficients are always recomputed from this source.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FastH3PddSchedule {
    grid_steps: usize,
    partition_indices: Vec<usize>,
    video_shift: f64,
    audio_shift: f64,
}
impl FastH3PddSchedule {
    pub fn new(
        grid_steps: usize,
        partition_indices: Vec<usize>,
        video_shift: f64,
        audio_shift: f64,
        max_steps: usize,
    ) -> Result<Self> {
        let schedule = Self {
            grid_steps,
            partition_indices,
            video_shift,
            audio_shift,
        };
        schedule.validate(max_steps)?;
        Ok(schedule)
    }
    pub fn validate(&self, max_steps: usize) -> Result<()> {
        ensure!(
            self.grid_steps >= 2 && self.grid_steps <= max_steps,
            "PDD fine grid exceeds capacity or has fewer than two intervals"
        );
        let grid_len = self
            .grid_steps
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("PDD grid extent overflows"))?;
        ensure!(
            grid_len <= isize::MAX as usize / std::mem::size_of::<f64>(),
            "PDD grid exceeds addressable storage"
        );
        ensure!(
            self.partition_indices.len() >= 2
                && self.partition_indices.len() <= grid_len
                && self.partition_indices.first() == Some(&0)
                && self.partition_indices.last() == Some(&self.grid_steps)
                && self
                    .partition_indices
                    .windows(2)
                    .all(|pair| pair[0] < pair[1]),
            "PDD partition must strictly increase from zero to grid_steps"
        );
        ensure!(
            [self.video_shift, self.audio_shift]
                .into_iter()
                .all(|s| s.is_finite() && s > 0.0),
            "PDD modality shifts must be finite and positive"
        );
        Ok(())
    }
    pub fn grid_steps(&self) -> usize {
        self.grid_steps
    }
    pub fn partition_indices(&self) -> &[usize] {
        &self.partition_indices
    }
    pub fn video_shift(&self) -> f64 {
        self.video_shift
    }
    pub fn audio_shift(&self) -> f64 {
        self.audio_shift
    }
}

/// A modality's coefficients for one fused PDD transformer evaluation.
#[derive(Clone, Debug)]
pub struct FastH3ModalityStep {
    sigma: f32,
    next_sigma: f32,
    timestep: f32,
    integration_weights: Vec<f64>,
    normalized_head_weights: Vec<f32>,
}

impl FastH3ModalityStep {
    pub fn sigma(&self) -> f32 {
        self.sigma
    }
    pub fn next_sigma(&self) -> f32 {
        self.next_sigma
    }
    pub fn timestep(&self) -> f32 {
        self.timestep
    }
    pub fn integration_weights(&self) -> &[f64] {
        &self.integration_weights
    }
    /// Normalize in f64, then cast to f32 for the fused head accumulation,
    /// matching PDDReplicatedLinear._fused_params. Projection storage precision
    /// is a separate checkpoint property; these coefficients do not quantize it.
    pub fn normalized_head_weights(&self) -> &[f32] {
        &self.normalized_head_weights
    }
}

#[derive(Clone, Debug)]
pub struct FastH3PddStep {
    pub heads: Range<usize>,
    pub video: FastH3ModalityStep,
    pub audio: FastH3ModalityStep,
}

/// Immutable trained schedule. Request progress belongs to the execution
/// runtime; sharing this plan never shares a mutable sampler cursor.
#[derive(Clone, Debug)]
pub struct FastH3PddPlan {
    schedule: FastH3PddSchedule,
    fine_grid: Vec<f64>,
    steps: Vec<FastH3PddStep>,
    video_head_width: usize,
    audio_head_width: usize,
}

impl FastH3PddPlan {
    pub fn new(checkpoint: &FastH3Checkpoint) -> Result<Self> {
        let schedule = FastH3PddSchedule::new(
            checkpoint.pdd_steps(),
            checkpoint.pdd_step_indices().to_vec(),
            checkpoint.video_shift(),
            checkpoint.audio_shift(),
            checkpoint.pdd_steps(),
        )?;
        Self::from_schedule(
            schedule,
            checkpoint.transformer().video_head_width(),
            checkpoint.transformer().audio_head_width(),
            checkpoint.pdd_steps(),
        )
    }
    /// Rebuild every clock and coefficient under an explicit fine-grid capacity.
    pub fn from_schedule(
        schedule: FastH3PddSchedule,
        video_head_width: usize,
        audio_head_width: usize,
        max_steps: usize,
    ) -> Result<Self> {
        schedule.validate(max_steps)?;
        let count = schedule.grid_steps;
        for width in [video_head_width, audio_head_width] {
            ensure!(
                width > 0
                    && width
                        .checked_mul(count)
                        .is_some_and(|n| n <= isize::MAX as usize),
                "PDD widened head extent is zero or unaddressable"
            );
        }
        ensure!(
            schedule.partition_indices.len() - 1
                <= isize::MAX as usize / std::mem::size_of::<FastH3PddStep>(),
            "PDD evaluation storage is unaddressable"
        );
        let step = FAST_H3_GRID_MAX_T / count as f64;
        // torch.linspace computes its second half from the endpoint. Preserve
        // that ordering rather than repeatedly subtracting a rounded step.
        let fine_grid: Vec<_> = (0..=count)
            .map(|index| {
                if index < (count + 1) / 2 {
                    FAST_H3_GRID_MAX_T - step * index as f64
                } else {
                    step * (count - index) as f64
                }
            })
            .collect();
        let mut steps = Vec::with_capacity(schedule.partition_indices.len() - 1);
        for pair in schedule.partition_indices.windows(2) {
            let heads = pair[0]..pair[1];
            steps.push(FastH3PddStep {
                video: modality_step(&fine_grid, heads.clone(), schedule.video_shift)?,
                audio: modality_step(&fine_grid, heads.clone(), schedule.audio_shift)?,
                heads,
            });
        }
        Ok(Self {
            schedule,
            fine_grid,
            steps,
            video_head_width,
            audio_head_width,
        })
    }
    pub fn schedule(&self) -> &FastH3PddSchedule {
        &self.schedule
    }
    pub fn fine_grid(&self) -> &[f64] {
        &self.fine_grid
    }
    pub fn steps(&self) -> &[FastH3PddStep] {
        &self.steps
    }
    pub fn video_head_width(&self) -> usize {
        self.video_head_width
    }
    pub fn audio_head_width(&self) -> usize {
        self.audio_head_width
    }
    /// Apply this only to target noise, never to retained reference latents.
    pub fn initial_video_noise_scale(&self) -> f32 {
        self.steps[0].video.sigma
    }
    pub fn initial_audio_noise_scale(&self) -> f32 {
        self.steps[0].audio.sigma
    }
}

#[cfg(test)]
mod tests;

fn sigma(base: f64, shift: f64) -> f64 {
    if shift == 1.0 {
        base
    } else {
        shift * base * FAST_H3_GRID_MAX_T / (base * (shift - 1.0) + FAST_H3_GRID_MAX_T)
    }
}

fn modality_step(grid: &[f64], heads: Range<usize>, shift: f64) -> Result<FastH3ModalityStep> {
    let integration_weights: Vec<_> = heads
        .clone()
        .map(|index| {
            let (a, b) = (grid[index], grid[index + 1]);
            if shift == 1.0 {
                b - a
            } else {
                let m = FAST_H3_GRID_MAX_T;
                shift * m.powi(2) * (b - a) / ((m + (shift - 1.0) * a) * (m + (shift - 1.0) * b))
            }
        })
        .collect();
    ensure!(
        integration_weights
            .iter()
            .all(|value| value.is_finite() && *value < 0.0),
        "PDD integration weights must be finite and negative"
    );
    let total: f64 = integration_weights.iter().sum();
    ensure!(total.is_finite() && total < 0.0, "invalid PDD block total");
    let normalized_head_weights: Vec<f32> = integration_weights
        .iter()
        .map(|value| (value / total) as f32)
        .collect();
    ensure!(
        normalized_head_weights
            .iter()
            .all(|value| value.is_finite() && *value > 0.0),
        "PDD normalized coefficients underflow or overflow"
    );
    let sigma = sigma(grid[heads.start], shift) as f32;
    let next_sigma = self::sigma(grid[heads.end], shift) as f32;
    ensure!(
        sigma.is_finite() && next_sigma.is_finite() && sigma > next_sigma && next_sigma >= 0.0,
        "PDD block sigmas collapse in f32"
    );
    Ok(FastH3ModalityStep {
        sigma,
        next_sigma,
        timestep: 1.0_f32 - sigma,
        integration_weights,
        normalized_head_weights,
    })
}
