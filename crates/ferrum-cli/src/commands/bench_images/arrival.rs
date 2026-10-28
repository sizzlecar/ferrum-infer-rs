//! Absolute arrival schedules. Client capacity never delays a due arrival:
//! callers record an unsent sample instead of silently slowing the workload.

use anyhow::{ensure, Context};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum)]
pub enum ImageArrivalMode {
    #[default]
    ClosedLoop,
    Burst,
    FixedInterval,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ImageArrivalPolicy {
    #[default]
    ClosedLoop,
    Burst,
    FixedInterval {
        interval_ms: f64,
    },
}

impl ImageArrivalPolicy {
    pub(super) fn from_cli(
        mode: ImageArrivalMode,
        interval_ms: Option<f64>,
    ) -> anyhow::Result<Self> {
        match (mode, interval_ms) {
            (ImageArrivalMode::ClosedLoop, None) => Ok(Self::ClosedLoop),
            (ImageArrivalMode::Burst, None) => Ok(Self::Burst),
            (ImageArrivalMode::FixedInterval, Some(interval_ms)) => {
                let policy = Self::FixedInterval { interval_ms };
                policy.validate(1)?;
                Ok(policy)
            }
            _ => anyhow::bail!(
                "--arrival-interval-ms is required only with --arrival-mode fixed-interval"
            ),
        }
    }

    pub(super) fn validate(self, sample_count: usize) -> anyhow::Result<()> {
        if let Self::FixedInterval { interval_ms } = self {
            let interval = Duration::try_from_secs_f64(interval_ms / 1000.0)
                .context("arrival interval must be finite, positive and representable")?;
            ensure!(
                !interval.is_zero(),
                "arrival interval must be at least one nanosecond"
            );
            let last = self
                .offset(sample_count.saturating_sub(1))
                .context("arrival schedule exceeds the clock range")?;
            Instant::now()
                .checked_add(last)
                .context("arrival schedule exceeds the clock range")?;
        }
        Ok(())
    }

    pub(super) fn offset(self, sample_index: usize) -> Option<Duration> {
        match self {
            Self::ClosedLoop => None,
            Self::Burst => Some(Duration::ZERO),
            Self::FixedInterval { interval_ms } => {
                Duration::try_from_secs_f64(interval_ms / 1000.0)
                    .ok()?
                    .checked_mul(u32::try_from(sample_index).ok()?)
            }
        }
    }

    pub(super) fn is_open_loop(self) -> bool {
        !matches!(self, Self::ClosedLoop)
    }
}

/// Clock evidence for every intended arrival, including client-capacity drops.
/// IDs align with request records; None means no HTTP send was initiated.
#[derive(Debug, Clone, Serialize)]
pub struct ImageArrivalRecord {
    pub sample_index: usize,
    pub phase: ferrum_bench_core::images::ImageRequestPhase,
    pub repetition: usize,
    pub scheduled_offset_ms: Option<f64>,
    pub dispatched_offset_ms: Option<f64>,
    /// Actual dispatch lateness, not a replacement for server queue latency.
    pub dispatch_lag_ms: Option<f64>,
}
